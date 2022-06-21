# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class ICIPRoIHead(StandardRoIHead):
    """ICIP RoI Head
    During training only considers RoI with IoU >= 0.5
    During testing uses BBoxes obtained by the RCNN head
    """

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_orientations,
                      # num_orientations=4,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_orientations (list[Tensor]): orientation labels for bbox head
            num_orientations (int): number of classes to divide the 360 degrees range in
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_orientations[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_orientations[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:

            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_orientations,
                                                    img_metas)
            losses.update(bbox_results['loss_pose'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        pose_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        """
        in our case, with 512 bboxes
        we obtain a 512, 256, 7, 7
        256 is out_channels of the roi_extractor
        7x7 is the outsize of RoIAlign
        256x7x7 = 12,544
        """

        if self.with_shared_head:
            pose_feats = self.shared_head(pose_feats)
        pose_feats = pose_feats.flatten(1)
        pose_score, pose_no_pred = self.bbox_head(pose_feats)

        bbox_results = dict(
            pose_score=pose_score, pose_no_pred=pose_no_pred, pose_feats=pose_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_orientations,
                            img_metas):
        """
        Run forward function and calculate loss for box head in training.
        TODO: We only want the positive bboxes to be counted for orientation loss
        """
        rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        # rois = bbox2roi([res.bboxes for res in sampling_results])
        pose_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_orientations, self.train_cfg)
        loss_pose = self.bbox_head.loss(pose_results['pose_score'],
                                        pose_results['pose_no_pred'], rois,
                                        *bbox_targets)

        pose_results.update(loss_pose=loss_pose)
        return pose_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_orientations)
            for i in range(len(det_bboxes))
        ]

        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        pose_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        pose_score = pose_results['pose_score']
        pose_no_pred = pose_results['pose_no_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        pose_score = pose_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if pose_no_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(pose_no_pred, torch.Tensor):
                pose_no_pred = pose_no_pred.split(num_proposals_per_img, 0)
            else:
                pose_no_pred = self.bbox_head.bbox_pred_split(
                    pose_no_pred, num_proposals_per_img)
        else:
            pose_no_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features)
                        # (0, self.bbox_head.fc_cls.out_features+1) # because they expect a background class
                    )

            else:
                # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # pose_score_augmented = torch.cat((pose_score[i], torch.zeros((pose_score[i].shape[0], 1)).to(device)), 1)
                # pose_score_augmented = torch.cat((pose_score[i], torch.empty((pose_score[i].shape[0], 1)).to(device)), 1)
                # get_bboxes softmaxes the output, so we can add a 0 without problems
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    # this is needed because get_bboxes expects a background class
                    pose_score[i],
                    # pose_score_augmented,
                    pose_no_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def async_simple_test(self,
                          x,
                          proposal_list,
                          img_metas,
                          proposals=None,
                          rescale=False):
        assert False, 'Called async_simple_test in ICIP_roi_head!'

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        assert False, 'Called aug_test in ICIP_roi_head!'