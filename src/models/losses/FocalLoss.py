import torch
import torch.nn as nn

# from torch.autograd import Variable


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(
        self,
        gamma=2,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        balance_param=0.25,
    ):
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        # weight = Variable(self.weight)
        loss_fn = nn.BCEWithLogitsLoss()
        logpt = loss_fn(input, target)
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss
