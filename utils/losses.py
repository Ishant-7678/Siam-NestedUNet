from utils.parser import get_parser_with_args
from utils.metrics import FocalLoss, dice_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)

    for prediction in predictions:

        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += bce + dice

    return loss
def relu_evidence(logits):
    """
    Convert logits into non-negative evidence.
    Softplus is preferred over ReLU for smoother gradients.
    """
    return F.softplus(logits)
def kl_divergence(alpha):
    """
    KL divergence between predicted Dirichlet and uniform Dirichlet prior.
    """

    device = alpha.device

    beta = torch.ones_like(alpha)

    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = (
        torch.lgamma(S_alpha)
        - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    )

    lnB_uni = (
        torch.sum(torch.lgamma(beta), dim=1, keepdim=True)
        - torch.lgamma(S_beta)
    )

    dg0 = torch.digamma(S_alpha)

    dg1 = torch.digamma(alpha)

    kl = (
        torch.sum(
            (alpha - beta) * (dg1 - dg0),
            dim=1,
            keepdim=True,
        )
        + lnB
        + lnB_uni
    )    

    return kl
def loglikelihood_loss(target, alpha):
    """
    Expected mean square error under Dirichlet distribution.
    """

    S = torch.sum(alpha, dim=1, keepdim=True)

    err = torch.sum((target - alpha / S) ** 2, dim=1, keepdim=True)

    var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)),
        dim=1,
        keepdim=True,
    )

    return (err + var)
def edl_loss(output, target, epoch_num, num_classes, annealing_step):
    """
    Sensoy Evidential Deep Learning Loss
    """

    evidence = relu_evidence(output)
    alpha = evidence + 1

    # Convert segmentation mask to one-hot
    if target.dim() == 4:
        target = target.squeeze(1)
    target = F.one_hot(target.long(), num_classes=num_classes)
    target = target.permute(0, 3, 1, 2).float()


    loss = loglikelihood_loss(target, alpha)
    annealing_coef = min(
        1.0,
        float(epoch_num + 1) / float(annealing_step)
    )

    kl_alpha = (alpha - 1) * (1 - target) + 1

    kl = annealing_coef * kl_divergence(kl_alpha)

    return (loss + kl).mean()
class SensoyEDLLoss(nn.Module):
    def __init__(self, num_classes=2, annealing_step=100):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step

    def forward(self, output, target, epoch_num):
        return edl_loss(
            output,
            target,
            epoch_num,
            self.num_classes,
            self.annealing_step
        )
