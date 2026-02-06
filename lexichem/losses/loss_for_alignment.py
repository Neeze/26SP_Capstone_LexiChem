import torch
from torch import nn
from torch.nn import functional as F

class VICRegLoss(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.lam = 25.0
        self.mu = 25.0
        self.nu = 1.0

    def forward(self, x, y):
        B, D = x.shape

        # 1. Invariance: MSE Loss
        sim_loss = F.mse_loss(x, y)

        # 2. Variance: Hinge loss std >= 1
        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)
        std_loss = F.relu(1 - std_x).mean() + F.relu(1 - std_y).mean()

        # 3. Covariance
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        cov_x = (x.T @ x) / (B - 1)
        cov_y = (y.T @ y) / (B - 1)
        cov_loss = ((cov_x.pow(2).sum() - cov_x.diag().pow(2).sum()) + 
                    (cov_y.pow(2).sum() - cov_y.diag().pow(2).sum())) / D

        return self.lam * sim_loss + self.mu * std_loss + self.nu * cov_loss


class InfoNCELoss(nn.Module):
    def __init__(self, args=None):
        super().__init__()

    def forward(self, x, y, logit_scale):
        B = x.size(0)
        logits = torch.matmul(x, y.t()) * logit_scale
        target = torch.arange(B, device=x.device)
        return F.cross_entropy(logits, target)


class NegativeCosineSimilarityLoss(nn.Module):
    def __init__(self, args=None):
        super().__init__()

    def forward(self, x, y):
        return -F.cosine_similarity(x, y, dim=-1).mean()