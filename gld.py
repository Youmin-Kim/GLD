import torch
from torch import nn
import torch.nn.functional as F


class GLDLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=500.0, spatial_size=8, div=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_local = div * div
        self.cross_entropy = nn.CrossEntropyLoss()
        self.t_local_pool = nn.AvgPool2d((spatial_size // div), stride=(spatial_size // div))
        self.s_local_pool = nn.AvgPool2d((spatial_size // div), stride=(spatial_size // div))
        self.t_global_pool = nn.AvgPool2d(spatial_size, stride=1)
        self.s_global_pool = nn.AvgPool2d(spatial_size, stride=1)

    def forward(self, t_f, s_f, t_fc, s_fc, target):
        t_global_pen = self.t_global_pool(t_f)
        t_global_pen = t_global_pen.view(t_global_pen.size(0), -1)
        t_global_logit = t_fc(t_global_pen)

        t_local_pen = self.t_local_pool(t_f)
        t_local_pen = t_local_pen.view(t_local_pen.size(0), t_local_pen.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        t_local_logit = t_fc(t_local_pen)

        s_global_pen = self.s_global_pool(s_f)
        s_global_pen = s_global_pen.view(s_global_pen.size(0), -1)
        s_global_logit = s_fc(s_global_pen)

        s_local_pen = self.s_local_pool(s_f)
        s_local_pen = s_local_pen.view(s_local_pen.size(0), s_local_pen.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        s_local_logit = s_fc(s_local_pen)

        t_logits = torch.cat((t_global_logit, t_local_logit), dim=0)
        s_logits = torch.cat((s_global_logit, s_local_logit), dim=0)

        task_loss = self.cross_entropy(s_global_logit, target)
        global_loss = F.kl_div(F.log_softmax(self.mean_var_norm(s_global_logit), dim=1), F.softmax(self.mean_var_norm(t_global_logit), dim=1),
                                 reduction='batchmean')
        local_loss = F.kl_div(F.log_softmax(self.mean_var_norm(s_local_logit), dim=1), F.softmax(self.mean_var_norm(t_local_logit), dim=1),
                               reduction='batchmean')
        relation_loss = self.dist_preserve_loss(t_logits, s_logits)
        distill_loss = self.alpha * global_loss + self.num_local * local_loss + self.beta * relation_loss

        return (1. - self.alpha) * task_loss, distill_loss

    def mean_var_norm(self, in_logit):
        norm_output = in_logit / in_logit.std(1).unsqueeze(1)
        return norm_output

    def dist_preserve_loss(self, t, s):
        bsz = s.size()[0]
        f_s = s.view(bsz, -1)
        f_t = t.view(bsz, -1)

        s_square = f_s.pow(2).sum(dim=1)
        s_prod = torch.mm(f_s, torch.t(f_s))
        G_s = (s_square.unsqueeze(1) + s_square.unsqueeze(0) - 2. * s_prod)
        G_s = torch.nn.functional.normalize(G_s)

        t_square = f_t.pow(2).sum(dim=1)
        t_prod = torch.mm(f_t, torch.t(f_t))
        G_t = (t_square.unsqueeze(1) + t_square.unsqueeze(0) - 2. * t_prod)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss
