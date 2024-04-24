import math

import torch
from torch import nn
import torch.nn.functional as F

import vs_helper


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.sqrt_d_k = math.sqrt(d_k)

    def forward(self, Q, K, V):
        attn = torch.bmm(Q, K.transpose(2, 1))
        attn = attn / self.sqrt_d_k

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        y = torch.bmm(attn, V)

        return y, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head=8, num_feature=1024):
        super().__init__()
        self.num_head = num_head

        self.Q = nn.Linear(num_feature, num_feature, bias=False)
        self.K = nn.Linear(num_feature, num_feature, bias=False)
        self.V = nn.Linear(num_feature, num_feature, bias=False)

        self.d_k = num_feature // num_head
        self.attention = ScaledDotProductAttention(self.d_k)

        self.fc = nn.Sequential(
            nn.Linear(num_feature, num_feature, bias=False),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        _, seq_len, num_feature = x.shape  # [1, seq_len, 1024]
        K = self.K(x)  # [1, seq_len, 1024]
        Q = self.Q(x)  # [1, seq_len, 1024]
        V = self.V(x)  # [1, seq_len, 1024]

        K = K.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len,
                                                                                              self.d_k)
        Q = Q.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len,
                                                                                              self.d_k)
        V = V.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len,
                                                                                              self.d_k)

        y, attn = self.attention(Q, K, V)  # [num_head, seq_len, d_k]
        y = y.view(1, self.num_head, seq_len, self.d_k).permute(0, 2, 1, 3).contiguous().view(1, seq_len, num_feature)

        y = self.fc(y)

        return y, attn


class AttentionExtractor(MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *inputs):
        out, _ = super().forward(*inputs)
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Reconstruction(nn.Module):
    def __init__(self, num_feature):
        super(Reconstruction, self).__init__()
        self.fc1 = nn.Linear(num_feature, num_feature * 2)
        self.fc2 = nn.Linear(num_feature * 2, num_feature)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, feature):
        hidden = self.lrelu(self.fc1(feature))
        out = self.fc2(hidden)
        out = self.sigmoid(out)
        return out


class STeMI(nn.Module):
    def __init__(self, num_feature, num_hidden, num_head, temporal_scales, spatial_scales):
        super().__init__()
        self.attention = AttentionExtractor(num_head, num_feature)
        self.layer_norm = nn.LayerNorm(num_feature)
        self.num_feature = num_feature
        self.num_hidden = num_hidden
        self.temporal_scales = temporal_scales
        self.spatial_scales = spatial_scales
        self.spatial_fc_1 = nn.Linear(num_feature, num_feature)
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, 1, 32))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, 1, 32))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, 1, 32))

        nn.init.trunc_normal_(self.pos_embed_1, std=.02)
        nn.init.trunc_normal_(self.pos_embed_2, std=.02)
        nn.init.trunc_normal_(self.pos_embed_3, std=.02)

        self.reconstruction = Reconstruction(num_feature)

        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
        )

        self.merge_extractor = nn.Sequential(
            nn.Linear(num_feature, num_feature),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.LayerNorm(num_feature)
        )

        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)
        self.fc_ctr = nn.Linear(num_hidden, 1)

    def forward(self, x, support_feature, support_target):
        support_target = support_target.squeeze(0)
        support_summary = support_feature[:, support_target, :]
        spatial_support_feature = support_feature.clone()
        spatial_support_feature += self.pos_embed_1.repeat(1, 32, 1).transpose(1, 2).reshape([1, 1, 1024])
        spatial_support_feature = self.spatial_fc_1(spatial_support_feature)
        spatial_support_summary = support_summary.clone()
        spatial_support_summary += self.pos_embed_2.repeat(1, 32, 1).transpose(1, 2).reshape([1, 1, 1024])
        spatial_support_summary = self.spatial_fc_1(spatial_support_summary)
        spatial_x = x.clone()
        spatial_x += self.pos_embed_3.repeat(1, 32, 1).transpose(1, 2).reshape([1, 1, 1024])
        spatial_x = self.spatial_fc_1(spatial_x)
        support_feat_out = spatial_support_feature.view(spatial_support_feature.shape[0], spatial_support_feature.shape[1], 32, 32)
        support_summary_out = spatial_support_summary.view(spatial_support_summary.shape[0], spatial_support_summary.shape[1], 32, 32)
        x_out = spatial_x.view(spatial_x.shape[0], spatial_x.shape[1], 32, 32)

        recon_support = support_feat_out.clone()
        recon_x = x_out.clone()

        merge_scales_space = []
        height = x_out.shape[3]
        for i in range(self.spatial_scales):
            if i > 0:
                height = int(height / 2)
                adapt_pool_sfo = nn.AdaptiveAvgPool2d((x_out.shape[2], height)).to(x.device)
                adapt_pool_sso = nn.AdaptiveAvgPool2d((x_out.shape[2], height)).to(x.device)
                adapt_pool_xot = nn.AdaptiveAvgPool2d((x_out.shape[2], height)).to(x.device)
                support_feat_out = adapt_pool_sfo(support_feat_out)
                support_summary_out = adapt_pool_sso(support_summary_out)
                x_out = adapt_pool_xot(x_out)
            merge_scale = torch.cat([support_feat_out, support_summary_out, x_out], 1)
            input_channels = support_feat_out.shape[1] + support_summary_out.shape[1] + x_out.shape[1]
            feature_compress = nn.Sequential(
                nn.Conv2d(input_channels, x_out.shape[1], kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ).to(x.device)
            compress_merge = feature_compress(merge_scale)
            merge_scales_space.append(compress_merge)
        merge_scales_all_space = torch.cat(merge_scales_space, 3)
        merge_scales_all_space = F.interpolate(merge_scales_all_space, size=(merge_scales_all_space.shape[2], merge_scales_all_space.shape[2]))
        merge_scales_all_space = merge_scales_all_space.view(merge_scales_all_space.shape[0], merge_scales_all_space.shape[1], 1024)

        support_feat_out = self.attention(support_feature)
        support_feat_out = support_feat_out + support_feature
        support_summary_out = self.attention(support_summary)
        support_summary_out = support_summary_out + support_summary
        support_strengthen = torch.bmm(support_feature, support_summary.transpose(1, 2))
        _, _, dim = support_strengthen.shape
        fc_1 = nn.Linear(dim, self.num_feature).to(x.device)
        support_updim = fc_1(support_strengthen)
        x_out = self.attention(x)
        x_out = x_out + x

        merge_scales_tpl = []
        row_sfo = support_feat_out.shape[1]
        row_sso = support_summary_out.shape[1]
        row_sup = support_updim.shape[1]
        row_xot = x_out.shape[1]
        column = support_feature.shape[2]
        for i in range(self.temporal_scales):
            adapt_pool_sfo = nn.AdaptiveAvgPool2d((row_sfo, column)).to(x.device)
            adapt_pool_sso = nn.AdaptiveAvgPool2d((row_sso, column)).to(x.device)
            adapt_pool_sup = nn.AdaptiveAvgPool2d((row_sup, column)).to(x.device)
            adapt_pool_xot = nn.AdaptiveAvgPool2d((row_xot, column)).to(x.device)
            sfo_scale = adapt_pool_sfo(support_feat_out).unsqueeze(0)
            sso_scale = adapt_pool_sso(support_summary_out).unsqueeze(0)
            sup_scale = adapt_pool_sup(support_updim).unsqueeze(0)
            xot_scale = adapt_pool_xot(x_out).unsqueeze(0)
            merge_scale = torch.cat([sfo_scale, sso_scale, sup_scale, xot_scale], 2)
            merge_scale = F.interpolate(merge_scale, size=(x.shape[1], x.shape[2]))
            merge_scales_tpl.append(merge_scale)
            row_sfo = int(row_sfo / 2)
            row_sso = int(row_sso / 2)
            row_sup = int(row_sup / 2)
            row_xot = int(row_xot / 2)
        merge_scales_tpl = torch.stack(merge_scales_tpl, dim=2)
        merge_scales_all_tpl = torch.mean(merge_scales_tpl, 2)
        merge_scales_all_tpl = merge_scales_all_tpl.squeeze(0)
        merge_scales_all = torch.cat([merge_scales_all_tpl, merge_scales_all_space], 2)    # 横向拼接
        _, _, dim_merge = merge_scales_all.shape
        fc_2 = nn.Linear(dim_merge, self.num_feature).to(x.device)
        merge_balance_dim = fc_2(merge_scales_all)
        merge_x = self.merge_extractor(merge_balance_dim)
        out = self.fc1(merge_x)

        _, seq_len, _ = x.shape
        pred_cls = self.fc_cls(out).sigmoid().view(seq_len)
        pred_loc = self.fc_loc(out).exp().view(seq_len, 2)
        pred_ctr = self.fc_ctr(out).sigmoid().view(seq_len)

        recon_x = recon_x.view(recon_x.shape[0], recon_x.shape[1], 1024)
        reconstruction_x = self.reconstruction(recon_x)
        recon_support = recon_support.view(recon_support.shape[0], recon_support.shape[1], 1024)
        reconstruction_support = self.reconstruction(recon_support)

        return pred_cls, pred_loc, pred_ctr, reconstruction_x, reconstruction_support

    def predict(self, seq, support_seq, support_summary):
        pred_cls, pred_loc, pred_ctr, _, _ = self(seq, support_seq, support_summary)
        pred_cls *= pred_ctr
        pred_cls /= pred_cls.max() + 1e-8
        pred_cls = pred_cls
        pred_loc = pred_loc
        pred_bboxes = vs_helper.offset2bbox(pred_loc)
        return pred_cls, pred_bboxes
