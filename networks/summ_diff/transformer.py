# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SummDiff Transformer with diffusion-based score denoising.

Based on DETR Transformer with modifications:
    * positional encodings are passed in MHattention
    * diffusion process for score prediction
    * supports DiT, Transformer decoder, and LatentMLP denoisers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
import numpy as np
from networks.summ_diff.attention import MultiheadAttention
from networks.summ_diff.latentmlp import SimpleMLP

from timm.models.vision_transformer import Mlp

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def exists(x):
    return x is not None   
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


class ScoreEmbed(nn.Module):
    def __init__(self, dim, K):
        super().__init__()
        self.dim = dim
        self.scale = 2 * math.pi
        self.K = K

    def forward(self, score):
        # Ensure input shape is (Batch_size, Sequence_length)
        B, L = score.shape
        
        # Normalize
        eps = 1e-6
        score = score / (self.K + eps) * self.scale
        
        dim_t = torch.arange(self.dim, dtype=torch.float32, device=score.device)
        dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.dim)
        embedding = score[:, :, None] / dim_t
        embedding = torch.stack((embedding[:, :, 0::2].cos(), -embedding[:, :, 1::2].sin()), dim=3).flatten(2)
        
        return embedding
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

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


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 K=200,
                 denoiser='DiT',
                 p_uncond=0.2,
                 w=0.1, sigmoid_temp=1.0, eps=1e-2,
                 scores_embed='learned',
                 ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers

        if denoiser == 'Transformer_dec':
            diff_layer = DiffDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        elif denoiser == 'DiT':
            diff_layer = DiffDecoderLayerDiT(d_model, nhead, dim_feedforward, dropout)
        elif denoiser == 'latentmlp':
            diff_layer = SimpleMLP(d_model, d_model, d_model, d_model, d_model, num_decoder_layers, 0.1, True, d_model)
        decoder_norm = nn.LayerNorm(d_model)
        self.diffusion_decoder = DiffDecoder(diff_layer, num_decoder_layers, decoder_norm,
                                          d_model=d_model, K=K, scores_embed=scores_embed,
                                          denoiser=denoiser)


        # build diffusion
        timesteps = 1000
        sampling_timesteps = 10
        # sampling_timesteps = 50 
        # self.num_proposals = num_queries

        betas = cosine_beta_schedule(timesteps)
        # betas =linear_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False

        self.scale =  1.0
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.diff_scores_embed = MLP(d_model, d_model, 1, 2)
        # self._reset_parameters()
        self.K = K # Quantization size for scores
        self.p_uncond = p_uncond
        self.w = w
        self.eps = eps
        self.sigmoid_temp = sigmoid_temp
        
    def inverse_sigmoid(self, x):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=self.eps)
        x2 = (1 - x).clamp(min=self.eps)
        return torch.log(x1/x2) / self.sigmoid_temp

    def sigmoid(self, x):
        x = self.sigmoid_temp * x
        return 1 / (1 + torch.exp(-x))

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self,tgt, x_x, batched_moments, time_cond,mask_local, pos_embed_local):
        # x_x = \hat{u_t}
        # x_scores = torch.clamp(x_x, min=-1 * self.scale, max=self.scale)
        # x_scores = ((x_scores / self.scale) + 1) / 2
        
        # x_scores = x_x.sigmoid()
        x_scores = self.sigmoid(x_x)
        ##########
        if self.p_uncond > 0:
            null_moments = batched_moments[:,-1:]
            batched_moments = batched_moments[:,:-1]
            mask_local = mask_local[:-1]
            pos_embed_local = pos_embed_local[:,:-1]
        ##########
        
        main_fea = self.diffusion_decoder(tgt, x_scores, batched_moments,
                            time_cond,memory_key_padding_mask=mask_local, pos=pos_embed_local)
        main_cood = self.diff_scores_embed(main_fea) 
        x_start = main_cood[-1].squeeze(-1)
        
        ###########
        if self.p_uncond > 0:
            null_fea = self.diffusion_decoder(tgt, x_scores, null_moments,
                            time_cond,memory_key_padding_mask=mask_local, pos=pos_embed_local)
            null_cood = self.diff_scores_embed(null_fea)
            x_start = (1 + self.w) * x_start - self.w * null_cood[-1].squeeze(-1)        
        ###########    
                
        # main_cood = main_cood.sigmoid()

        # x_start = main_cood[-1].squeeze(-1)  
        # x_start = (x_start * 2 - 1.) * self.scale
        # x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)

        pred_noise = self.predict_noise_from_start(x_x, time_cond, x_start)

        return pred_noise, x_start, main_fea, main_cood

    @torch.no_grad()
    def ddim_sample(self, tgt, batched_moments, memory_key_padding_mask, pos): # 
        L, B, D = batched_moments.size()
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(0, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x_start = None
        if self.p_uncond > 0:
            B -= 1
            #########
            # null_moments = batched_moments[:,-1:]
            # batched_moments = batched_moments[:,:-1]
            # memory_key_padding_mask = memory_key_padding_mask[:-1]
            # pos = pos[:,:-1]
            #########
        
        x_scores = torch.randn((B, L)).cuda()  

        # summe generation stability
        # x_scores = torch.clamp(x_scores, min=-1 * self.scale, max=self.scale)
        
        step_num = 0

        for time, time_next in time_pairs: #T-1->T-2 .....

            step_num = step_num+1

            time_cond = torch.full((B,), time).cuda().long()

            pred_noise, pred_start, hs, hs_cood = self.model_predictions(tgt, x_scores, batched_moments, 
                                                        time_cond, memory_key_padding_mask, pos)
            ##########
            # if self.p_uncond > 0:
            #     null_noise, null_start, null_hs, null_hs_cood = self.model_predictions(tgt, x_scores, null_moments, 
            #                                             time_cond, memory_key_padding_mask, pos)
            #     pred_noise = (1 + self.w) * pred_noise - self.w * null_noise
            #     pred_start = (1 + self.w) * pred_start - self.w * null_start
            ##########
            # pred_start [-1, 1] / hs hidden / hs_cood [0, 1]

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            # sigma = 0
            cc = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(pred_start)   
            x_scores = pred_start * alpha_next.sqrt() + cc * pred_noise + sigma * noise
            # divide x_scores by maximum absolute value to prevent explosion
            # x_scores = x_scores / x_scores.abs().max()

        hs_cood = self.diff_scores_embed(hs)
        # hs_cood = hs_cood.sigmoid()
        hs_cood = self.sigmoid(hs_cood)
        # return hs_class, hs_cood, hs
        return hs_cood, hs
            
    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.triu(noise)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise



    def prepare_diffusion_concat(self, gtscore, vid_length):
        """
        param gt_boxes: (center, long)
        """
        time = torch.randint(0, self.num_timesteps, (1,)).long().cuda()
        noise = torch.randn(len(gtscore)).cuda()  
        # padded = gtscore.shape[0]
        # if padded > vid_length: # gtscore is padded with zeros
        #     # pad noise with zeros to length padded
        #     noise = F.pad(noise, (0, padded - vid_length), value=0)
        # noise after vid_length is zero
        # noise = noise[:vid_length]
        # noise = F.pad(noise, (0, gtscore.shape[0] - len(noise)), value=0)

        # x_start = (gtscore * 2. - 1.) * self.scale # 
        # x_start = torch.clamp(gtscore, min=1 / (self.K), max=1 - 1 / (self.K))
        x_start = self.inverse_sigmoid(gtscore)
        # noise sample
        x_t = self.q_sample(x_start=x_start, t=time, noise=noise)
        # x_t = torch.clamp(x_t, min= -1 * self.scale, max= self.scale)
        # x_t = ((x_t / self.scale) + 1) / 2.
        # x_t = x_t.sigmoid()
        x_t = self.sigmoid(x_t)
        x_t = x_t[:vid_length]
        x_t = F.pad(x_t, (0, gtscore.shape[0] - len(x_t)), value=0)

        return x_t, noise, time


    def prepare_targets(self, targets, video_length):
        diffused_scores = []
        noises = []
        ts = []
        for gtscore, vid_length in zip(targets, video_length):
            d_score, d_noise, d_t = self.prepare_diffusion_concat(gtscore, vid_length)
            diffused_scores.append(d_score)
            noises.append(d_noise)
            ts.append(d_t)
        return torch.stack(diffused_scores),  torch.stack(noises), torch.stack(ts)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, gtscore, video_length=None):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            pos_embed: (batch_size, L, d) the same as src
            gtscore: (batch_size, L)

        Returns:

        """
        # flatten NxCxHxW to HWxNxC
        bs, l, d = src.shape
        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed = pos_embed.permute(1, 0, 2)   # (L, batch_size, d)
        # txt_src = src[video_length + 1:]

        # src = self.t2v_encoder(src, src_key_padding_mask=mask, pos=pos_embed, video_length=video_length)  # (L, batch_size, d)  # cross-attention
        # src = src[:video_length + 1] #torch.Size([76, 32, 256]
        # mask = mask[:, :video_length + 1]
        # pos_embed = pos_embed[:video_length + 1]
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # (L, batch_size, d)
        memory_global, memory_local = memory[0], memory[1:] # memory[0] is global token
        mask_local = mask[:, 1:] 
        pos_embed_local = pos_embed[1:]

        tgt = torch.zeros(bs, l, d).cuda()  

        if gtscore is not None:
            x_t_scores, noises, time_t = self.prepare_targets(gtscore, video_length)  # x_t_scores (B, L), noises (B, L)
            time_t = time_t.squeeze(-1)  # (B,)
            hs = self.diffusion_decoder(tgt, x_t_scores, memory_local, time_t, memory_key_padding_mask=mask_local, pos=pos_embed_local)
            # hs_class = self.diff_class_embed(hs) 
            hs_scores = self.diff_scores_embed(hs) 
            # outputs_scores = hs_scores.sigmoid()
            outputs_scores = self.sigmoid(hs_scores)
        else:
            outputs_scores, hs = self.ddim_sample(tgt, memory_local, memory_key_padding_mask=mask_local, pos=pos_embed_local)

        memory_local = memory_local.transpose(0, 1)  # (batch_size, L, d) condition

        # return hs_class, outputs_scores, hs, memory_local, memory_global
        return outputs_scores, hs, memory_local, memory_global

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        output = src

        intermediate = []

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, **kwargs)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]

        
        src2=torch.where(torch.isnan(src2),torch.full_like(src2,0),src2)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class DiffDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, query_dim=2, keep_query_pos=False,
                 modulate_t_attn=False,
                 K = 200,
                 denoiser='DiT',
                 scores_embed='learned',
                 ):
        super().__init__()
        if denoiser == 'Transformer_dec':
            self.layers = _get_clones(decoder_layer, num_layers)
        elif denoiser == 'DiT':
            self.layers = decoder_layer
        elif denoiser == 'latentmlp':
            self.layers = decoder_layer
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.d_model = d_model

        self.time_mlp = TimestepEmbedder(d_model)
        nn.init.normal_(self.time_mlp.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_mlp.mlp[2].weight, std=0.02)
        
        
        # self.noise_embed = MLP(query_dim, d_model, d_model, 1)
        if scores_embed == 'learned':
            self.noise_embed = nn.Embedding(K + 1, d_model)
            self.noise_map = MLP(d_model, d_model, d_model, 2)
            nn.init.normal_(self.noise_map.layers[0].weight, std=0.02)
            nn.init.normal_(self.noise_map.layers[1].weight, std=0.02)
        
        if scores_embed == 'sinusoidal':
            self.noise_embed = nn.Sequential(
                ScoreEmbed(d_model, K),
                # SinusoidalPositionEmbeddings_score(d_model),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model, bias=True),
            )   
            nn.init.normal_(self.noise_embed[1].weight, std=0.02)
            nn.init.normal_(self.noise_embed[3].weight, std=0.02)
            
        self.K = K
        self.denoiser_type = denoiser
        self.score_embed = scores_embed
        self.pos_emb = MLP(d_model, d_model, d_model, 2)
        nn.init.normal_(self.pos_emb.layers[0].weight, std=0.02)
        nn.init.normal_(self.pos_emb.layers[1].weight, std=0.02)
                        
                        
    def forward(self, tgt, scores, memory, time, 
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                ):
        
        # scores shape (B, L) 0 - 1
        # memory shape (L, B, D) Feature
        # time shape (B, ) 
        # pos shape (L, B, D) pos_embed_local
        
        L, B, D = memory.shape
        time_emb = self.time_mlp(time) # B, d
        
        # scores = inverse_sigmoid(scores).sigmoid()
        indices = torch.round(scores*(self.K)) # 0-1 -> 1-200
        indices = indices.long()
        output = self.noise_embed(indices) # B, L, D
        if self.score_embed == 'learned':
            output = self.noise_map(output) # B, L, D
        if self.denoiser_type == 'Transformer_dec':
            output += 0.1 * time_emb.unsqueeze(1).repeat(1,L,1)
        output = output.transpose(0, 1)  # (B, L, D) -> (L, B, D)
        intermediate = []
        
        if self.denoiser_type == 'Transformer_dec':
            for layer_id, layer in enumerate(self.layers):
                output = layer(output, memory, time_emb, tgt_mask=tgt_mask, # output (B, L), memory (L, B, D), time_emb (B, L)
                            memory_mask=memory_mask,  # None
                            tgt_key_padding_mask=tgt_key_padding_mask,  # None
                            memory_key_padding_mask=memory_key_padding_mask, # memory mask (B, L)
                            pos=pos, # query_pos=query_pos, query_sine_embed=query_sine_embed, # pos = pos_embed_local (L, B, D)
                            is_first=(layer_id == 0))
                if self.return_intermediate:
                    intermediate.append(self.norm(output))
                    
        elif self.denoiser_type == 'DiT':
            # pos = self.pos_emb(pos)
            output = self.layers(output, memory, 0.1 * time_emb, # output (B, L), memory (L, B, D), time_emb (B, L)
                        mask=memory_key_padding_mask, # memory mask (B, L)
                        pos=pos)
            
        elif self.denoiser_type == 'latentmlp':
            ###
            # pos = self.pos_emb(pos)
            ###
            output += pos
            output = self.layers(output, time, memory)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output) #2 layers

        if self.return_intermediate:
            return torch.stack(intermediate).transpose(1, 2) # (num_layers, B, L, D)
        return output.unsqueeze(0).transpose(1, 2) # (1, B, L, D)

class DiffDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = MLP(d_model, d_model, d_model, 3)

        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,time_emb,  # tgt: score (B, L) / memory: feature (L, B, D) / time_emb: time (B, L)
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,  # memory mask (B, L)
                pos: Optional[Tensor] = None, # (L, B, D)
                # query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False):

        L, bs, n_model = tgt.shape[0], tgt.shape[1], tgt.shape[2]
        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)  # Q
        k_content = self.ca_kcontent_proj(memory)  # K
        v = self.ca_v_proj(memory)  # V

        # B, L = q_content.shape
        # L, B, D = k_content.shape

        # Mapping scenario
        # k_pos = self.ca_kpos_proj(pos)

        if is_first:
            q = q_content + pos
            k = k_content
        else:
            q = q_content
            k = k_content

        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=v, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def modulate(x, shift, scale):
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_m = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.attn = MultiheadAttention(hidden_size, num_heads, dropout=dropout, vdim=hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = dim_feedforward
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)
        self.mlp = MLP(hidden_size, mlp_hidden_dim, hidden_size, 2)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, mem, c, mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        q = modulate(self.norm1(x), shift_msa, scale_msa)
        kv = modulate(self.norm_m(mem), shift_msa, scale_msa)
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        
        x = x + gate_msa * self.attn(query=q, key=k, value=v, key_padding_mask=mask)[0]
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiffDecoderLayerDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1,
        depth=2,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(d_model)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        # nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, memory, t, mask, pos):
        """
        Forward pass of DiT.
        x: (L, B, D) tensor of u_t
        t: (B, D) tensor of diffusion timestep embeddings
        memory: (L, B, D) tensor of video conditions
        pos: (L, B, D) tensor of positional embeddings
        mask: (B, L) tensor of padding mask
        """
        L, B, D = x.shape
        i = 0
        ###########
        x += pos
        ###########
        # memory += pos
        for block in self.blocks:
            if i == 0:
                c = t.unsqueeze(0).repeat(L, 1, 1)  + pos
                x = block(x, memory, c, mask)
            else:
                c = t.unsqueeze(0).repeat(L, 1, 1)  + pos
                x = block(x, memory, c, mask)
            i += 1
            # x = block(x, memory, c, mask)
        x = self.final_layer(x, t.unsqueeze(0).repeat(L, 1, 1))    # (N, T, patch_size ** 2 * out_channels)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        # num_queries=args.num_queries,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=False,
        activation='prelu',
        K=args.K,
        denoiser=args.denoiser,
        p_uncond=args.p_uncond,
        w=args.w,
        sigmoid_temp=args.sigmoid_temp,
        eps=args.eps,
        scores_embed=args.scores_embed,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
