import torch
from torch import nn
import torch.nn.functional as F
class ZINBLoss(nn.Module):
    def __init__(self, ridge_lambda=0):
        super(ZINBLoss, self).__init__()
        self.ridge_lambda = ridge_lambda

    def forward(self, x, mean, disp, pi, scale_factor):
        eps = 1e-10
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8 * torch.ones_like(x)), zero_case, nb_case)

        if self.ridge_lambda > 0:
            ridge = self.ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)

        return result


class ClusterLoss(nn.Module):

    def __init__(self):
        super(ClusterLoss, self).__init__()

    def forward(self, target, pred):
        return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=-1))


class ELOBkldLoss(nn.Module):
    def __init__(self):
        super(ELOBkldLoss, self).__init__()

    def forward(self, mu, logvar):
        result = -((0.5 * logvar) - (torch.exp(logvar) + mu ** 2) / 2. + 0.5)
        result = result.sum(dim=1).mean()

        return result


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature


    def forward(self, features):
        n_samples = features.shape[0]
        labels = torch.zeros(n_samples, dtype=torch.long, device=features.device)
        similarity_matrix = F.cosine_similarity(features[:, None], features[None, :], dim=2) / self.temperature
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss





