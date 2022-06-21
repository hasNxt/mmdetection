
from .convfc_bbox_head import ConvFCBBoxHead
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy

@HEADS.register_module()
class SigmaBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(SigmaBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.log_var_cls = nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.log_var_reg = nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.fc_cls.register_parameter('log_var_cls', self.log_var_cls)
        self.fc_reg.register_parameter('log_var_reg', self.log_var_reg)



    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    precision = torch.exp(-self.fc_cls.get_parameter('log_var_cls'))
                    precision = torch.pow(precision, 2)
                    weight_loss = torch.sum(precision * loss_cls_ + self.fc_cls.get_parameter('log_var_cls'), -1)
                    losses['loss_cls'] = weight_loss
                    losses['log_var_bbox_cls'] = self.fc_cls.get_parameter('log_var_cls')
                    losses['bbox_cls_precision'] = precision
                    # losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                loss_bbox = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
                reg_precision = torch.exp(-self.fc_reg.get_parameter('log_var_reg'))
                reg_precision = torch.pow(reg_precision, 2)
                reg_precision = torch.mul(reg_precision, 1/2)
                reg_weight_loss = torch.sum(reg_precision * loss_bbox + self.fc_reg.get_parameter('log_var_reg'), -1)
                losses['loss_bbox'] = reg_weight_loss
                losses['log_var_bbox_reg'] = self.fc_reg.get_parameter('log_var_reg')
                losses['bbox_reg_precision'] = reg_precision
                # losses['loss_bbox'] = self.loss_bbox(
                #     pos_bbox_pred,
                #     bbox_targets[pos_inds.type(torch.bool)],
                #     bbox_weights[pos_inds.type(torch.bool)],
                #     avg_factor=bbox_targets.size(0),
                #     reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses