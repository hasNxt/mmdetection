# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# from mmcls.models.losses import Accuracy
from ..builder import HEADS, build_loss, build_roi_extractor
from ..utils import is_tracing
from .base_head import BaseHead
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler


@HEADS.register_module()
class OrientationHead(BaseHead):
    """classification head. 2 FC layers + classification layer
    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use Mixup/CutMix or something like that during training,
            it is not reasonable to calculate accuracy. Defaults to False.
        train_cfg (dict): Config for training
        test_cfg (dict): Config for testing
        in_channels (int | list): width of fc layers e.g. [256, 128] without counting final classification layer
        num_classes (int): number of classes to identify
    """

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, ),
                 cal_acc=False,
                 train_cfg=None,
                 test_cfg=None,
                 in_channels=None,
                 num_classes=2,
                 num_orientations=2,
                 init_cfg=None,
                 pretrained=False
                 ):
        super(OrientationHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(loss, dict)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_loss = build_loss(loss)
        # self.compute_accuracy = Accuracy(topk=self.topk)
        # self.cal_acc = cal_acc
        self.cal_acc = False

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert in_channels is not None, 'in_channels not defined'
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_orientations = num_orientations

        self.bbox_roi_extractor = None
        self.bbox_assigner = None
        self.bbox_sampler = None

        self.init_assigner_sampler()

        # let's build the actual network
        self.model = self.build_layers()



    def init_assigner_sampler(self):
        """
        Initialize assigner and sampler from cfgs
        """
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_roi_extractor = build_roi_extractor(self.train_cfg.bbox_roi_extractor)
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def build_layers(self):
        """
        Builds fully connected layers
        """
        model = nn.Sequential(
            nn.Linear(self.in_channels[0], self.in_channels[1]),
            nn.ReLU(),
            nn.Linear(self.in_channels[1], self.num_orientations),
            nn.Softmax()
        )
        return model

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_orientations, gt_masks , **kwargs):
        """
        Pass through the assigner, only keeping proposals that have IoU > 0.5 with GT bboxes AND are person class
        For each proposal left, calculate orientation (quantized using self.num_classes)
        Compute loss

        we are going to use standard_roi_head for inspiration

        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposal_list (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_orientations (list[int]): orientations for each gt_bboxe
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        num_imgs = len(img_metas)
        gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                # gt_labels[i]
                gt_orientations[i]
            )
            """
            Attributes of assign_result:
            num_gts (int): the number of truth boxes considered when computing this
                assignment

            gt_inds (LongTensor): for each predicted box indicates the 1-based
                index of the assigned truth box. 0 means unassigned and -1 means
                ignore.

            max_overlaps (FloatTensor): the iou between the predicted box and its
                assigned truth box.

            labels (None | LongTensor): If specified, for each predicted box
                indicates the category label of the assigned truth box.
            """
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                # gt_labels[i],
                gt_orientations[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
            # what format does sampling_result have?
            """
            Attributes of sampling_result:
            self = <SamplingResult({
                'neg_bboxes': torch.Size([12, 4]),
                'neg_inds': tensor([ 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
                'num_gts': 4,
                'pos_assigned_gt_inds': tensor([], dtype=torch.int64),
                'pos_bboxes': torch.Size([0, 4]),
                'pos_inds': tensor([], dtype=torch.int64),
                'pos_is_gt': tensor([], dtype=torch.uint8)
            })>
            """

        losses = dict()
        # at this point we could iterate over the positive
        rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        # run features through self.model,
        # compile loss
        # return loss

        ori_preds = self.model(bbox_feats)
        # for each image, for each prediction (in softmax) select the index of the highest value
        img_res = []
        for img_preds in ori_preds:
            ori_res = []
            for pred in img_preds:
                np_pred = np.array(pred)
                ori_res.append(np.argmax(np_pred))
            img_res.append(ori_res)
        # probably need to convert img_res to tensor
        losses = self.loss(img_res, gt_orientations, **kwargs)
        return losses


    def loss(self, cls_score, gt_label, **kwargs):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(
            cls_score, gt_label, avg_factor=num_samples, **kwargs)
        # if self.cal_acc:
        #     # compute accuracy
        #     acc = self.compute_accuracy(cls_score, gt_label)
        #     assert len(acc) == len(self.topk)
        #     losses['accuracy'] = {
        #         f'top-{k}': a
        #         for k, a in zip(self.topk, acc)
        #     }
        losses['orientation_loss'] = loss
        return losses



    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]

        warnings.warn(
            'The input of ClsHead should be already logits. '
            'Please modify the backbone if you want to get pre-logits feature.'
        )
        return x

    def simple_test(self, cls_score, softmax=True, post_process=True):
        """Inference without augmentation.
        Args:
            cls_score (tuple[Tensor]): The input classification score logits.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.
        Returns:
            Tensor | list: The inference results.
                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred