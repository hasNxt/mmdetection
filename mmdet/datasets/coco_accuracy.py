# Copyright (c) OpenMMLab. All rights reserved.

"""
Dataset used with Faster R-CNN.
Each class is the correct orientation class
"""

import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset

import torch
from mmdet.core.bbox.assigners import MaxIoUAssigner
from sklearn.metrics import confusion_matrix, classification_report


@DATASETS.register_module()
class CocoDatasetAccuracy(CustomDataset):

    CLASSES = ('[-180,-90]',
            '(-90,0]',
            '(0,90]',
            '(90,180]')

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast', 'pose_acc']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            # if metric == 'pose_acc':
            #     eval_results['pose_acc'] = self.get_accuracy(results)
            #     print(f'\nPose_acc = {eval_results["pose_acc"]}\n')
            #     continue

            if metric == 'pose_acc':
                (pose_acc, pose_acc_noBBox, classwise_acc, y_true, y_pred) = self.get_accuracy(results)
                eval_results['pose_acc'] = pose_acc
                eval_results['pose_acc_noBBox'] = pose_acc_noBBox
                print(f'\nPose_Accuracy = {eval_results["pose_acc"]}')
                print(f'Pose_Accuracy_noMisses = {eval_results["pose_acc_noBBox"]}\n')
                print(f'Classwise Accuracy:')
                avg_rec = 0
                for k in classwise_acc:
                    # print(f'Class {k}:\n\tPose_Accuracy: {classwise_acc[k]["accuracy"]}'
                    #       f'\n\tPose_Accuracy_noMisses: {classwise_acc[k]["noMiss_accuracy"]}'
                    #       f'\n\tInstances: {classwise_acc[k]["denominator"]}'
                    #       f'\n\tPrecision: {classwise_acc[k]["precision"]}'
                    #       f'\n\tRecall: {classwise_acc[k]["recall"]}'
                    #       f'\n\tF1: {classwise_acc[k]["f1"]}'
                    #       )
                    eval_results[f'Class {k} Recall'] = classwise_acc[k]["recall"]
                    eval_results[f'Class {k} Precision'] = classwise_acc[k]["precision"]
                    avg_rec += classwise_acc[k]['recall']
                avg_rec = avg_rec / 4
                acc_rec = (pose_acc + avg_rec) / 2
                # print('\n')
                eval_results['Acc_Rec score'] = acc_rec
                # print('\n')

                labels = [str(i) for i in range(4)]
                labels.append('NotDet')

                # compute confusion matrix
                # conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
                # cmat = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
                # cmat.plot(cmap=plt.cm.Blues)
                # plt.savefig('CM_ICIP_TC4_sequential_480_weighted.png')


                class_report = classification_report(y_true, y_pred, target_names=labels)
                print(class_report)
                continue

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results



    # def get_accuracy(self, results):
    #
    #     bbox_threshold = 0.5
    #     gt_overlap_threshold = 0.7
    #
    #     # img_ids = self.img_ids
    #     numerator = 0
    #     denominator = 0
    #     if len(results) == 0:
    #         # accuracy is 0 if no results
    #         return 0
    #     cocoGT = self.coco
    #
    #     for idx, img_det in enumerate(results):
    #         det_bboxs = []
    #         det_labels = []
    #         gt_bboxs = []
    #         gt_labels = []
    #         # get a list of detections and their labels
    #         # new version returns list of orientations, not list of 1 item
    #         # for det_ori, bboxs_list in enumerate(img_det[0]):
    #         for det_ori, bboxs_list in enumerate(img_det):
    #             for bbox in bboxs_list:
    #                 if bbox[4] >= bbox_threshold:
    #                     # convert xyxy to xywh
    #                     tmp_bbox = bbox[:4]
    #                     # tmp_bbox[2] = tmp_bbox[2] - tmp_bbox[0]
    #                     # tmp_bbox[3] = tmp_bbox[3] - tmp_bbox[1]
    #                     det_bboxs.append(tmp_bbox)
    #                     # det_bboxs.append(bbox[:4])
    #                     det_labels.append(det_ori)
    #
    #         # get a list of gt_bboxs and their labels
    #         img_ann_ids = cocoGT.getAnnIds(imgIds=[self.img_ids[idx]])
    #         img_anns = cocoGT.loadAnns(img_ann_ids)
    #         for ann in img_anns:
    #             # need to convert the bbox from xywh to xyxy
    #             tmp_bbox = ann['bbox'].copy()
    #             tmp_bbox[2] = tmp_bbox[2] + tmp_bbox[0]
    #             tmp_bbox[3] = tmp_bbox[3] + tmp_bbox[1]
    #             gt_bboxs.append(tmp_bbox)
    #             gt_labels.append(ann['category_id'])
    #
    #         # now we need to map each detected bbox to a gt_bbox
    #         if len(det_labels) == 0:
    #             # if there are no bboxes skip
    #             denominator += len(gt_bboxs)
    #             continue
    #         assigner = MaxIoUAssigner(gt_overlap_threshold, gt_overlap_threshold, match_low_quality=False)
    #         t_det_bboxs = torch.Tensor(det_bboxs)
    #         t_gt_bboxs = torch.Tensor(gt_bboxs)
    #
    #         assign_result = assigner.assign(t_det_bboxs, t_gt_bboxs)
    #         # assign_result = assign_result.detach().cpu().numpy()
    #         gt_inds = assign_result.gt_inds.detach().cpu().numpy()
    #
    #         # create a list mapping gt_bboxes to detected ones
    #         gt_to_det = [[] for _ in gt_bboxs]
    #         # iterate on gt_inds, meaning detected bboxes, and add them to the corrisponding map slot, with their orientation
    #         for dt_idx, gt_idx in enumerate(gt_inds):
    #             gt_idx = gt_idx-1
    #             if gt_idx is None or gt_idx == len(gt_bboxs) or gt_idx == -1:
    #                 # if this detected bbox could not be mapped to anything, then it is a missed classification?
    #                 # technically we don't need to count for this, as mAP already keeps track of these things
    #                 # denominator += 1
    #                 denominator += 0
    #             else:
    #                 gt_to_det[gt_idx].append(dict(bbox=det_bboxs[dt_idx], orientation=det_labels[dt_idx]))
    #
    #         # calculate accuracy as (TP+TN)/(TP+TN+FP+FN)
    #         # important: missed bboxes count as wrong detections!
    #         for gt_bbox_idx, gt_bbox in enumerate(gt_to_det):
    #             if len(gt_bbox) == 0:
    #                 denominator += 1
    #             else:
    #                 true_ori = int(gt_labels[gt_bbox_idx])
    #                 for mapped in gt_bbox:
    #                     denominator += 1
    #                     if mapped['orientation'] == true_ori:
    #                         numerator += 1
    #
    #     # return accuracy
    #     return float(numerator)/float(denominator)

    def get_accuracy(self, results):

        bbox_threshold = 0.5
        gt_overlap_threshold = 0.7

        # img_ids = self.img_ids
        numerator = 0
        denominator = 0
        noMiss_denominator = 0
        if len(results) == 0:
            # accuracy is 0 if no results
            return 0
        cocoGT = self.coco
        num_orientations = len(results[0])
        classwise_accuracy = {}
        for i in range(num_orientations):
            classwise_accuracy[i] = dict(numerator=0, denominator=0, noMiss_denominator=0, accuracy=0,
                                         noMiss_accuracy=0, tp=0, tn=0, fp=0, fn=0, precision=0, recall=0)

        # for confusion matrix
        # labels include an additional class to accomodate for missed bboxes
        labels = [str(i) for i in range(num_orientations)]
        labels.append('NotDet')
        y_true = []
        y_pred = []
        # y_true.append(num_orientations)
        # y_pred.append(num_orientations)
        for idx, img_det in enumerate(results):
            det_bboxs = []
            det_labels = []
            gt_bboxs = []
            gt_labels = []
            # get a list of detections and their labels
            # new version returns list of orientations, not list of 1 item
            # for det_ori, bboxs_list in enumerate(img_det[0]):
            for det_ori, bboxs_list in enumerate(img_det):
                for bbox in bboxs_list:
                    if bbox[4] >= bbox_threshold:
                        # convert xyxy to xywh
                        tmp_bbox = bbox[:4]
                        # tmp_bbox[2] = tmp_bbox[2] - tmp_bbox[0]
                        # tmp_bbox[3] = tmp_bbox[3] - tmp_bbox[1]
                        det_bboxs.append(tmp_bbox)
                        # det_bboxs.append(bbox[:4])
                        det_labels.append(det_ori)

            # get a list of gt_bboxs and their labels
            img_ann_ids = cocoGT.getAnnIds(imgIds=[self.img_ids[idx]])
            img_anns = cocoGT.loadAnns(img_ann_ids)
            for ann in img_anns:
                # need to convert the bbox from xywh to xyxy
                tmp_bbox = ann['bbox'].copy()
                tmp_bbox[2] = tmp_bbox[2] + tmp_bbox[0]
                tmp_bbox[3] = tmp_bbox[3] + tmp_bbox[1]
                gt_bboxs.append(tmp_bbox)
                gt_labels.append(ann['category_id'])

            # now we need to map each detected bbox to a gt_bbox
            if len(det_labels) == 0:
                # if there are no bboxes skip
                for i in range(len(gt_bboxs)):
                    classwise_accuracy[int(gt_labels[i])]['denominator'] += 1
                    y_pred.append(num_orientations)
                    y_true.append(int(gt_labels[i]))
                denominator += len(gt_bboxs)
                continue
            if len(gt_bboxs) == 0:
                continue
            assigner = MaxIoUAssigner(gt_overlap_threshold, gt_overlap_threshold, match_low_quality=False)
            t_det_bboxs = torch.Tensor(det_bboxs)
            t_gt_bboxs = torch.Tensor(gt_bboxs)

            assign_result = assigner.assign(t_det_bboxs, t_gt_bboxs)
            # assign_result = assign_result.detach().cpu().numpy()
            gt_inds = assign_result.gt_inds.detach().cpu().numpy()

            # create a list mapping gt_bboxes to detected ones
            gt_to_det = [[] for _ in gt_bboxs]
            # iterate on gt_inds, meaning detected bboxes, and add them to the corrisponding map slot, with their orientation
            for dt_idx, gt_idx in enumerate(gt_inds):
                gt_idx = gt_idx-1
                if gt_idx is None or gt_idx == len(gt_bboxs) or gt_idx == -1:
                    # if this detected bbox could not be mapped to anything, then it is a missed classification?
                    # technically we don't need to count for this, as mAP already keeps track of these things
                    # denominator += 1
                    denominator += 0
                else:
                    gt_to_det[gt_idx].append(dict(bbox=det_bboxs[dt_idx], orientation=det_labels[dt_idx]))

            # calculate accuracy as (TP+TN)/(TP+TN+FP+FN)
            # important: missed bboxes count as wrong detections!
            for gt_bbox_idx, gt_bbox in enumerate(gt_to_det):
                if len(gt_bbox) == 0:
                    true_ori = int(gt_labels[gt_bbox_idx])
                    classwise_accuracy[true_ori]['fn'] += 1
                    classwise_accuracy[true_ori]['denominator'] += 1
                    denominator += 1
                    # for matrix
                    y_pred.append(num_orientations)
                    y_true.append(true_ori)
                else:
                    true_ori = int(gt_labels[gt_bbox_idx])
                    for mapped in gt_bbox:
                        classwise_accuracy[true_ori]['denominator'] += 1
                        classwise_accuracy[true_ori]['noMiss_denominator'] += 1
                        denominator += 1
                        noMiss_denominator += 1
                        # for matrix
                        y_true.append(true_ori)
                        y_pred.append(mapped['orientation'])
                        if mapped['orientation'] == true_ori:
                            classwise_accuracy[true_ori]['numerator'] += 1
                            classwise_accuracy[true_ori]['tp'] += 1
                            for k in classwise_accuracy:
                                if k != true_ori:
                                    classwise_accuracy[k]['tn'] += 1
                            numerator += 1
                        else:
                            classwise_accuracy[true_ori]['fn'] += 1
                            classwise_accuracy[mapped['orientation']]['fp'] += 1


        # return accuracy
        for k in classwise_accuracy:
            # precision
            if classwise_accuracy[k]['fp']+classwise_accuracy[k]['tp'] == 0:
                classwise_accuracy[k]['precision'] = 1
            else:
                classwise_accuracy[k]['precision'] = float(classwise_accuracy[k]['tp'])/(float(classwise_accuracy[k]['fp']+classwise_accuracy[k]['tp']))

            # recall
            if classwise_accuracy[k]['fn'] + classwise_accuracy[k]['tp'] == 0:
                classwise_accuracy[k]['recall'] = 1
            else:
                classwise_accuracy[k]['recall'] = float(classwise_accuracy[k]['tp']) / (
                    float(classwise_accuracy[k]['fn'] + classwise_accuracy[k]['tp']))

            # f1_score
            if float(classwise_accuracy[k]['precision'] + classwise_accuracy[k]['recall']) > 0:
                classwise_accuracy[k]['f1'] = float(
                    2 * (classwise_accuracy[k]['precision'] * classwise_accuracy[k]['recall'])) / float(
                    classwise_accuracy[k]['precision'] + classwise_accuracy[k]['recall'])
            else:
                classwise_accuracy[k]['f1'] = 0

            if classwise_accuracy[k]['denominator'] == 0:
                classwise_accuracy[k]['accuracy'] = 1
            else:
                classwise_accuracy[k]['accuracy'] = float(classwise_accuracy[k]['numerator'])/float(classwise_accuracy[k]['denominator'])

            if classwise_accuracy[k]['noMiss_denominator'] == 0:
                classwise_accuracy[k]['noMiss_accuracy'] = 1
            else:
                classwise_accuracy[k]['noMiss_accuracy'] = float(classwise_accuracy[k]['numerator'])/float(classwise_accuracy[k]['noMiss_denominator'])

        acc = 0
        acc_noMiss = 0
        if denominator != 0:
            acc = float(numerator) / float(denominator)
        if noMiss_denominator != 0:
            acc_noMiss = float(numerator) / float(noMiss_denominator)
        return (
            acc, acc_noMiss, classwise_accuracy, y_true, y_pred)