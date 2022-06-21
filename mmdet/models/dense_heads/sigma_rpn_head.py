# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms

from ..builder import HEADS
from .anchor_head import AnchorHead


@HEADS.register_module()
class SigmaRPNHead(AnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        num_convs (int): Number of convolution layers in the head. Default 1.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 num_convs=1,
                 **kwargs):
        self.num_convs = num_convs
        super(SigmaRPNHead, self).__init__(
            1, in_channels, init_cfg=init_cfg, **kwargs)
        self.log_var_cls = nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.log_var_reg = nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.rpn_cls.register_parameter('log_var_cls', self.log_var_cls)
        self.rpn_reg.register_parameter('log_var_reg', self.log_var_reg)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4,
                                 1)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred


    # def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
    #                 bbox_targets, bbox_weights, num_total_samples):
    #     """Compute loss of a single scale level.
    #
    #     Args:
    #         cls_score (Tensor): Box scores for each scale level
    #             Has shape (N, num_anchors * num_classes, H, W).
    #         bbox_pred (Tensor): Box energies / deltas for each scale
    #             level with shape (N, num_anchors * 4, H, W).
    #         anchors (Tensor): Box reference for each scale level with shape
    #             (N, num_total_anchors, 4).
    #         labels (Tensor): Labels of each anchors with shape
    #             (N, num_total_anchors).
    #         label_weights (Tensor): Label weights of each anchor with shape
    #             (N, num_total_anchors)
    #         bbox_targets (Tensor): BBox regression targets of each anchor
    #             weight shape (N, num_total_anchors, 4).
    #         bbox_weights (Tensor): BBox regression loss weights of each anchor
    #             with shape (N, num_total_anchors, 4).
    #         num_total_samples (int): If sampling, num total samples equal to
    #             the number of total anchors; Otherwise, it is the number of
    #             positive anchors.
    #
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     # classification loss
    #     labels = labels.reshape(-1)
    #     label_weights = label_weights.reshape(-1)
    #     cls_score = cls_score.permute(0, 2, 3,
    #                                   1).reshape(-1, self.cls_out_channels)
    #
    #     loss_cls = self.loss_cls(
    #         cls_score, labels, label_weights, avg_factor=num_total_samples)
    #
    #     precision = torch.exp(-self.rpn_cls.get_parameter('log_var_cls'))
    #     precision = torch.pow(precision, 2)
    #     weight_loss = torch.sum(precision * loss_cls + self.rpn_cls.get_parameter('log_var_cls'), -1)
    #
    #
    #     # regression loss
    #     bbox_targets = bbox_targets.reshape(-1, 4)
    #     bbox_weights = bbox_weights.reshape(-1, 4)
    #     bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
    #     if self.reg_decoded_bbox:
    #         # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
    #         # is applied directly on the decoded bounding boxes, it
    #         # decodes the already encoded coordinates to absolute format.
    #         anchors = anchors.reshape(-1, 4)
    #         bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
    #     loss_bbox = self.loss_bbox(
    #         bbox_pred,
    #         bbox_targets,
    #         bbox_weights,
    #         avg_factor=num_total_samples)
    #     reg_precision = torch.exp(-self.rpn_reg.get_parameter('log_var_reg'))
    #     reg_precision = torch.pow(reg_precision, 2)
    #     reg_precision = torch.mul(reg_precision, 1 / 2)
    #     reg_weight_loss = torch.sum(reg_precision * loss_bbox + self.rpn_reg.get_parameter('log_var_reg'), -1)
    #     return weight_loss, reg_weight_loss
    #



    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super(SigmaRPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)


        precision = torch.exp(-self.rpn_cls.get_parameter('log_var_cls'))
        precision = torch.pow(precision, 2)
        # rpn_log_var_cls = self.rpn_cls.get_parameter('log_var_cls')
        # rpn_cls_precision = precision


        reg_precision = torch.exp(-self.rpn_reg.get_parameter('log_var_reg'))
        reg_precision = torch.pow(reg_precision, 2)
        reg_precision = torch.mul(reg_precision, 1 / 2)
        # rpn_reg_precision = reg_precision
        # rpn_log_var_reg = self.rpn_reg.get_parameter('log_var_reg')

        losses_rpn_cls = losses['loss_cls']
        losses_rpn_bbox = losses['loss_bbox']

        # classification
        loss_rpn_cls = sum(_loss.mean() for _loss in losses_rpn_cls)
        loss_rpn_cls = torch.sum(precision*loss_rpn_cls + self.rpn_cls.get_parameter('log_var_cls'), -1)

        # regression
        loss_rpn_bbox = sum(_loss.mean() for _loss in losses_rpn_bbox)

        loss_rpn_bbox = torch.sum(reg_precision * loss_rpn_bbox + self.rpn_reg.get_parameter('log_var_reg'), -1)

        return dict(
            loss_rpn_cls=loss_rpn_cls, loss_rpn_bbox=loss_rpn_bbox,
            rpn_log_var_cls=self.rpn_cls.get_parameter('log_var_cls'), rpn_cls_precision=precision,
            rpn_log_var_reg=self.rpn_reg.get_parameter('log_var_reg'), rpn_reg_precision=reg_precision
        )

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_anchors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has
                shape (num_anchors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RPN head does not need this value.
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_anchors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']

        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            anchors = mlvl_anchors[level_idx]
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        return self._bbox_post_process(mlvl_scores, mlvl_bbox_preds,
                                       mlvl_valid_anchors, level_ids, cfg,
                                       img_shape)

    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors,
                           level_ids, cfg, img_shape, **kwargs):
        """bbox post-processing method.

        Do the nms operation for bboxes in same level.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            mlvl_valid_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_bboxes, 4).
            level_ids (list[Tensor]): Indexes from all scale levels of a
                single image, each item has shape (num_bboxes, ).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, `self.test_cfg` would be used.
            img_shape (tuple(int)): The shape of model's input image.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() > 0:
            dets, _ = batched_nms(proposals, scores, ids, cfg.nms)
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg.max_per_img]

    def onnx_export(self, x, img_metas):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.
        Returns:
            Tensor: dets of shape [N, num_det, 5].
        """
        cls_scores, bbox_preds = self(x)

        assert len(cls_scores) == len(bbox_preds)

        batch_bboxes, batch_scores = super(SigmaRPNHead, self).onnx_export(
            cls_scores, bbox_preds, img_metas=img_metas, with_nms=False)
        # Use ONNX::NonMaxSuppression in deployment
        from mmdet.core.export import add_dummy_nms_for_onnx
        cfg = copy.deepcopy(self.test_cfg)
        score_threshold = cfg.nms.get('score_thr', 0.0)
        nms_pre = cfg.get('deploy_nms_pre', -1)
        # Different from the normal forward doing NMS level by level,
        # we do NMS across all levels when exporting ONNX.
        dets, _ = add_dummy_nms_for_onnx(batch_bboxes, batch_scores,
                                         cfg.max_per_img,
                                         cfg.nms.iou_threshold,
                                         score_threshold, nms_pre,
                                         cfg.max_per_img)
        return dets
