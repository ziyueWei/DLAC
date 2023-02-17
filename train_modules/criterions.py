#!/usr/bin/env python
# coding: utf-8

import torch
from helper.utils import get_hierarchy_relations
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.5, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        pt = torch.sigmoid(logits)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * labels * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - labels) * torch.log(1 - pt)
        return torch.mean(loss)
        
class ClassificationLoss(torch.nn.Module):
    def __init__(self,
                 taxonomic_hierarchy,
                 label_map,
                 recursive_penalty,
                 recursive_constraint=True, loss_type="bce"):
        """
        Criterion class, classfication loss & recursive regularization
        :param taxonomic_hierarchy:  Str, file path of hierarchy taxonomy
        :param label_map: Dict, label to id
        :param recursive_penalty: Float, lambda value <- config.train.loss.recursive_regularization.penalty
        :param recursive_constraint: Boolean <- config.train.loss.recursive_regularization.flag
        """
        super(ClassificationLoss, self).__init__()
        self.loss_type = loss_type
        self.focal_loss_fn = FocalLoss()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.recursive_relation = get_hierarchy_relations(taxonomic_hierarchy,
                                                          label_map)
        self.recursive_penalty = recursive_penalty
        self.recursive_constraint = recursive_constraint

    def _recursive_regularization(self, params, device):
        """
        recursive regularization: constraint on the parameters of classifier among parent and children
        :param params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
        :param device: torch.device -> config.train.device_setting.device
        :return: loss -> torch.FloatTensor, ()
        """
        rec_reg = 0.0
        for i in range(len(params)):
            if i not in self.recursive_relation.keys():
                continue
            child_list = self.recursive_relation[i]
            if not child_list:
                continue
            child_list = torch.tensor(child_list).to(device)
            child_params = torch.index_select(params, 0, child_list)
            parent_params = torch.index_select(params, 0, torch.tensor(i).to(device))
            parent_params = parent_params.repeat(child_params.shape[0], 1)
            _diff = parent_params - child_params
            diff = _diff.view(_diff.shape[0], -1)
            rec_reg += 1.0 / 2 * torch.norm(diff, p=2) ** 2
        return rec_reg

    def forward(self, logits, targets, recursive_params):
        """
        :param logits: torch.FloatTensor, (batch, N)
        :param targets: torch.FloatTensor, (batch, N)
        :param recursive_params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
        """
        device = logits.device
        if self.loss_type == "bce":
            loss = self.loss_fn(logits, targets)
        elif self.loss_type == "focal":
            loss = self.focal_loss_fn(logits, targets)
        else:
            loss = 0
        if self.recursive_constraint:
            loss += self.loss_fn(logits, targets) + \
                   self.recursive_penalty * self._recursive_regularization(recursive_params,
                                                                           device)
        return loss

class MarginRankingLoss(torch.nn.Module):
    def __init__(self, config):
        """
        Criterion loss
        default torch.nn.MarginRankingLoss(0.01)
        """
        super(MarginRankingLoss, self).__init__()
        self.dataset = config.data.dataset
        base = 0.2
        self.ranking = [torch.nn.MarginRankingLoss(margin=base*0.1), torch.nn.MarginRankingLoss(margin=base * 0.5),
                        torch.nn.MarginRankingLoss(margin=base)]
        self.negative_ratio = config.data.negative_ratio


    def forward(self, text_repre, label_repre_positive, label_repre_negative, mask=None):
        """
        :param text_repre: torch.FloatTensor, (batch, hidden)
        :param label_repre_positive: torch.FloatTensor, (batch, hidden)
        :param label_repre_negative: torch.FloatTensor, (batch, sample_num, hidden)
        :param mask: torch.BoolTensor, (batch, negative_ratio, negative_number), the index of different label
        """
        loss_inter_total, loss_intra_total = 0, 0

        text_score = text_repre.unsqueeze(1).repeat(1, label_repre_positive.size(1), 1)
        loss_inter = (torch.pow(text_score - label_repre_positive, 2)).sum(-1)
        loss_inter = F.relu(loss_inter / text_repre.size(-1))
        loss_inter_total += loss_inter.mean()

        for i in range(self.negative_ratio):
            m = mask[:, i]
            m = m.unsqueeze(-1).repeat(1, 1, label_repre_negative.size(-1))
            label_n_score = torch.masked_select(label_repre_negative, m)
            label_n_score = label_n_score.view(text_repre.size(0), -1, label_repre_negative.size(-1))
            text_score = text_repre.unsqueeze(1).repeat(1, label_n_score.size(1), 1)

            # index 0: parent node
            if i == 0:
                loss_inter_parent = (torch.pow(text_score - label_n_score, 2)).sum(-1)
                loss_inter_parent = F.relu((loss_inter_parent-0.01) / text_repre.size(-1))
                loss_inter_total += loss_inter_parent.mean()
            else:
                # index 1: wrong sibling, index 2: other wrong label
                loss_intra = (torch.pow(text_score - label_n_score, 2)).sum(-1)
                loss_intra = F.relu(loss_intra / text_repre.size(-1))
                loss_gold = loss_inter.view(1, -1)
                loss_cand = loss_intra.view(1, -1)
                ones = torch.ones(loss_gold.size()).to(loss_gold.device)
                loss_intra_total += self.ranking[i](loss_gold, loss_cand, ones)
        return loss_inter_total, loss_intra_total

# wzy的添加:新的损失函数
class ResampleLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True, partial=False,
                 loss_weight=1.0, reduction='mean',
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 focal=dict(
                     focal=True,
                     alpha=0.5,
                     gamma=2,
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 class_freq=None,
                 train_num=None):
        super(ResampleLoss, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid  # true
        self.partial = partial  # false
        self.loss_weight = loss_weight  # 1.0
        self.reduction = reduction  # mean
        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        # reweighting function
        self.reweight_func = reweight_func  # rebalance

        # normalization (optional)
        self.weight_norm = weight_norm # none

        # focal loss params
        self.focal = focal['focal']  # true
        self.gamma = focal['gamma']  # 2
        self.alpha = focal['alpha']  # 0.5  change to alpha

        # mapping function params
        self.map_alpha = map_param['alpha'] # 0.1
        self.map_beta = map_param['beta']  # 10
        self.map_gamma = map_param['gamma']  # 0.9

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']  # 0.9
        self.CB_mode = CB_loss['CB_mode']  # average_w

        # self.class_freq: 每个标签的词频
        a = []
        for i in class_freq.values():
            a.append(i)
        # self.class_freq = a
        self.class_freq = torch.from_numpy(np.asarray(a)).float().cuda()
        # self.num_classes: 标签总数
        self.num_classes = self.class_freq.shape[0]
        # self.train_num: 训练集总数
        self.train_num = train_num # only used to be divided by class_freq
        # regularization params
        # self.logit_reg:  init_bias:0.05 neg_scale:2.0
        self.logit_reg = logit_reg
        # self.neg_scale: 2.0
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias ########################## bug fixed https://github.com/wutong16/DistributionBalancedLoss/issues/8

        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)  # mean

        weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        if self.focal:
            logpt = self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(-logpt)
            wtloss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            # alpha_t = torch.where(label==1, self.alpha, 1-self.alpha)
            alpha_t = 0.5
            loss = alpha_t * ((1 - pt) ** self.gamma) * wtloss ####################### balance_param should be a tensor
            loss = reduce_loss(loss, reduction)             ############################ add reduction
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def logit_reg_functions(self, labels, logits, weight=None):
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale  + logits * labels
            if weight is not None:
                weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if  'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
                    torch.sum(gt_labels, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, avg_n)).cuda()
        elif 'average_w' in self.CB_mode:
            weight_ = torch.tensor((1 - self.CB_beta)).cuda() / \
                      (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
                     torch.sum(gt_labels, dim=1, keepdim=True)
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, min_n)).cuda()
        else:
            raise NameError
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
        return weight

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss