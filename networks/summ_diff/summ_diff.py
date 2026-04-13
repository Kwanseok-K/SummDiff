"""
SummDiff model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from networks.summ_diff.transformer import build_transformer
from networks.summ_diff.position_encoding import build_position_encoding
import numpy as np


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class SummDiff(nn.Module):
    """ SummDiff: Diffusion-based Video Summarization. """

    def __init__(self, transformer, position_embed, vid_dim,
                 num_scores, input_dropout, n_input_proj=2):
        super().__init__()
        self.transformer = transformer
        self.position_embed = position_embed
        hidden_dim = transformer.d_model
        self.n_input_proj = n_input_proj
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

    def forward(self, gtscore, src_vid, src_vid_mask, n_frames):
        """
        Args:
            gtscore: [batch_size, L_vid] ground truth scores (None during inference)
            src_vid: [batch_size, L_vid, D_vid] video features
            src_vid_mask: [batch_size, L_vid] containing True on valid frames
            n_frames: list of frame counts per video
        Returns:
            dict with 'pred_scores', and a placeholder weight (0.0)
        """
        src_vid = self.input_vid_proj(src_vid)
        pos_vid = self.position_embed(src_vid, src_vid_mask)

        # Prepend global representation token
        mask_ = torch.tensor([[True]]).to(src_vid_mask.device).repeat(src_vid_mask.shape[0], 1)
        mask = torch.cat([mask_, src_vid_mask], dim=1)
        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src_vid.shape[0], 1, 1)
        src = torch.cat([src_, src_vid], dim=1)
        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos_vid.shape[0], 1, 1)
        pos = torch.cat([pos_, pos_vid], dim=1)

        outputs_scores, hs, memory, memory_global = self.transformer(src, ~mask, pos, gtscore, video_length=n_frames)
        out = {'pred_scores': outputs_scores[-1].squeeze()}

        return out, 0.0


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding = build_position_encoding(args)

    model = SummDiff(
        transformer,
        position_embedding,
        vid_dim=args.v_feat_dim,
        num_scores=args.num_scores,
        input_dropout=args.input_dropout,
        n_input_proj=args.n_input_proj,
    )

    criterion = SummDiffCriterion(dec_loss_coef=args.dec_loss_coef, aux_loss_coef=args.aux_loss_coef)
    criterion.to(device)
    return model, criterion


class SummDiffCriterion(nn.Module):
    def __init__(self, dec_loss_coef=1, aux_loss_coef=1):
        super().__init__()
        self.dec_loss_coef = dec_loss_coef
        self.aux_loss_coef = aux_loss_coef

    def forward(self, outputs, gtscores, gt_summary, mask):
        loss = 0
        if "pred_scores" in outputs:
            if len(outputs["pred_scores"].shape) == 1:
                outputs["pred_scores"] = outputs["pred_scores"].unsqueeze(0)
            loss += self.dec_loss_coef * F.mse_loss(outputs["pred_scores"][mask], gtscores[mask])
        return loss
