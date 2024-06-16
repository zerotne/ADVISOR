import torch
from torch import nn
from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import SelfAttentionLayer,\
    CrossAttentionLayer, FFNLayer, MLP, _get_activation_fn
from scipy.optimize import linear_sum_assignment


import torch
import torch.nn as nn
import torch.nn.functional as F

class DisturbanceFilter(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        kernel_size=3  # for temporal convolution
    ):
        super().__init__()
        
        self.temporal_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _get_activation_fn(activation)(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        indentify,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        # Temporal Convolution Block
        tgt2 = self.temporal_conv(tgt.transpose(1, 2)).transpose(1, 2)
        tgt = self.norm1(tgt + self.dropout(tgt2))

        # Multi-head Attention
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = self.norm2(tgt + self.dropout(tgt2))

        # Self-Attention
        tgt2 = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt
        )[0]
        tgt = self.norm3(tgt + self.dropout(tgt2))

        # Feedforward Neural Network
        tgt2 = self.feedforward(tgt)
        tgt = self.norm4(tgt + self.dropout(tgt2))

        return tgt

    def forward_pre(
        self,
        indentify,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        # Temporal Convolution Block
        tgt2 = self.temporal_conv(tgt.transpose(1, 2)).transpose(1, 2)
        tgt2 = self.norm1(tgt + self.dropout(tgt2))

        # Multi-head Attention
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt2 = self.norm2(tgt2 + self.dropout(tgt2))

        # Self-Attention
        tgt2 = self.self_attn(
            query=tgt2,
            key=tgt2,
            value=tgt2
        )[0]
        tgt2 = self.norm3(tgt2 + self.dropout(tgt2))

        # Feedforward Neural Network
        tgt2 = self.feedforward(tgt2)
        tgt = self.norm4(indentify + self.dropout(tgt2))

        return tgt

    def forward(self, indentify, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(indentify, tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(indentify, tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)



class ProgressiveTracker(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        class_num=25,
    ):
        super(ProgressiveTracker, self).__init__()

        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                DisturbanceFilter(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_channel)

        # init heads
        self.class_embed = nn.Linear(hidden_channel, class_num + 1)
        self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)

        # record previous frame information
        self.last_outputs = None
        self.last_frame_embeds = None

    def _clear_memory(self):
        del self.last_outputs
        self.last_outputs = None
        return

    def forward(self, frame_embeds, mask_features, resume=False, return_indices=False):
        
        frame_embeds = frame_embeds.permute(2, 3, 0, 1)  # t, q, b, c
        n_frame, n_q, bs, _ = frame_embeds.size()
        outputs = []
        ret_indices = []

        for i in range(n_frame):
            ms_output = []
            single_frame_embeds = frame_embeds[i]  # q b c
            # the first frame of a video
            if i == 0 and resume is False:
                self._clear_memory()
                self.last_frame_embeds = single_frame_embeds
                for j in range(self.num_layers):
                    if j == 0:
                        ms_output.append(single_frame_embeds)
                        ret_indices.append(self.match_embds(single_frame_embeds, single_frame_embeds))
                        output = self.transformer_cross_attention_layers[j](
                            single_frame_embeds, single_frame_embeds, single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers[j](
                            ms_output[-1], ms_output[-1], single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
            else:
                for j in range(self.num_layers):
                    if j == 0:
                        ms_output.append(single_frame_embeds)
                        indices = self.match_embds(self.last_frame_embeds, single_frame_embeds)
                        self.last_frame_embeds = single_frame_embeds[indices]
                        ret_indices.append(indices)
                        output = self.transformer_cross_attention_layers[j](
                            single_frame_embeds[indices], self.last_outputs[-1], single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers[j](
                            ms_output[-1], self.last_outputs[-1], single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
            ms_output = torch.stack(ms_output, dim=0)  # (1 + layers, q, b, c)
            self.last_outputs = ms_output
            outputs.append(ms_output[1:])
        outputs = torch.stack(outputs, dim=0)  # (t, l, q, b, c)
        outputs_class, outputs_masks = self.prediction(outputs, mask_features)
        outputs = self.decoder_norm(outputs)
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
           'pred_masks': outputs_masks[-1],  # (b, q, t, h, w)
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks
           ),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1)  # (b, c, t, q)
        }
        if return_indices:
            return out, ret_indices
        else:
            return out

    def match_embds(self, ref_embds, cur_embds):
        #  embeds (q, b, c)
        ref_embds, cur_embds = ref_embds.detach()[:, 0, :], cur_embds.detach()[:, 0, :]
        ref_embds = ref_embds / (ref_embds.norm(dim=1)[:, None] + 1e-6)
        cur_embds = cur_embds / (cur_embds.norm(dim=1)[:, None] + 1e-6)
        cos_sim = torch.mm(ref_embds, cur_embds.transpose(0, 1))
        C = 1 - cos_sim

        C = C.cpu()
        C = torch.where(torch.isnan(C), torch.full_like(C, 0), C)

        indices = linear_sum_assignment(C.transpose(0, 1))
        indices = indices[1]
        return indices

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a.transpose(1, 2), "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]

    def prediction(self, outputs, mask_features):
        # outputs (t, l, q, b, c)
        # mask_features (b, t, c, h, w)
        decoder_output = self.decoder_norm(outputs)
        decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
        outputs_class = self.class_embed(decoder_output).transpose(2, 3)  # (l, b, q, t, cls+1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)
        return outputs_class, outputs_mask

    def frame_forward(self, frame_embeds):
        
        bs, n_channel, n_frame, n_q = frame_embeds.size()
        frame_embeds = frame_embeds.permute(3, 0, 2, 1)  # (q, b, t, c)
        frame_embeds = frame_embeds.flatten(1, 2)  # (q, bt, c)

        for j in range(self.num_layers):
            if j == 0:
                output = self.transformer_cross_attention_layers[j](
                    frame_embeds, frame_embeds, frame_embeds,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_self_attention_layers[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers[j](
                    output
                )
            else:
                output = self.transformer_cross_attention_layers[j](
                    output, output, frame_embeds,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_self_attention_layers[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers[j](
                    output
                )
        output = self.decoder_norm(output)
        output = output.reshape(n_q, bs, n_frame, n_channel)
        return output.permute(1, 3, 2, 0)




class EnhancedLocalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(EnhancedLocalSelfAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class RefinementCompensator(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, num_repeats=1):
        super(RefinementCompensator, self).__init__()
        self.enhanced_local_self_attn = EnhancedLocalSelfAttention(d_model, num_heads, dropout)
        self.self_attention = SelfAttentionLayer(d_model=d_model, nhead=num_heads, dropout=dropout, normalize_before=False)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.ffn = FFNLayer(d_model=d_model, dim_feedforward=d_ff, dropout=dropout, normalize_before=False)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.num_repeats = num_repeats

    def forward(self, x):
        
        x = self.enhanced_local_self_attn(x)
        
        for _ in range(self.num_repeats):
            
            x = self.add_norm1(x, self.self_attention)
            
            x = self.add_norm2(x, self.ffn)

        return x





class InteractionAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(InteractionAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        attn_output, _ = self.multihead_attn(tgt, tgt, tgt)
        return self.norm(tgt + self.dropout(attn_output))

class AsymmetricConvNetwork(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super(AsymmetricConvNetwork, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.norm(x.transpose(1, 2))



class SpatialInteractionModule(nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        class_num=25,
        windows=5
    ):
        super(SpatialInteractionModule, self).__init__()

        self.windows = windows

        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.cross_attention_layers = nn.ModuleList()
        self.interaction_attention_layers = nn.ModuleList()
        self.asymmetric_conv_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.sigmoid_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.1
                )
            )

            self.interaction_attention_layers.append(
                InteractionAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.1
                )
            )

            self.asymmetric_conv_layers.append(
                AsymmetricConvNetwork(
                    d_model=hidden_channel,
                    kernel_size=3
                )
            )

            self.ffn_layers.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.1
                )
            )

            self.sigmoid_layers.append(nn.Sigmoid())

        self.decoder_norm = nn.LayerNorm(hidden_channel)

        self.class_embed = nn.Linear(hidden_channel, class_num + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, mask_dim)
        )

        self.activation_proj = nn.Linear(hidden_channel, 1)

    def forward(self, instance_embeds, frame_embeds, mask_features):
        n_batch, n_channel, n_frames, n_instance = instance_embeds.size()

        outputs = []
        output = instance_embeds
        frame_embeds = frame_embeds.permute(3, 0, 2, 1).flatten(1, 2)

        for i in range(self.num_layers):
            output = output.permute(2, 0, 3, 1).flatten(1, 2)

            output = self.cross_attention_layers[i](
                output, frame_embeds, frame_embeds
            )

            output = output.permute(1, 2, 0)
            output = self.asymmetric_conv_layers[i](output).transpose(1, 2).reshape(
                n_batch, n_instance, n_channel, n_frames
            ).permute(1, 0, 3, 2).flatten(1, 2)

            output = self.interaction_attention_layers[i](output)
            output = self.ffn_layers[i](output)

            output = self.sigmoid_layers[i](output)
            output = output.reshape(n_instance, n_batch, n_frames, n_channel).permute(1, 3, 2, 0)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0).permute(3, 0, 4, 1, 2)
        outputs_class, outputs_masks = self.prediction(outputs, mask_features)
        outputs = self.decoder_norm(outputs)
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),
           'pred_masks': outputs_masks[-1],
           'aux_outputs': self._set_aux_loss(outputs_class, outputs_masks),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1)
        }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        return [{"pred_logits": a.transpose(1, 2), "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]

    def windows_prediction(self, outputs, mask_features, windows=5):
        iters = outputs.size(0) // windows
        if outputs.size(0) % windows != 0:
            iters += 1
        outputs_classes = []
        outputs_masks = []
        for i in range(iters):
            start_idx = i * windows
            end_idx = (i + 1) * windows
            clip_outputs = outputs[start_idx:end_idx]
            decoder_output = self.decoder_norm(clip_outputs)
            decoder_output = decoder_output.permute(1, 3, 0, 2, 4)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum(
                "lbtqc,btchw->lbqthw",
                mask_embed,
                mask_features[:, start_idx:end_idx].to(mask_embed.device)
            )
            outputs_classes.append(decoder_output)
            outputs_masks.append(outputs_mask.cpu().to(torch.float32))
        outputs_classes = torch.cat(outputs_classes, dim=2)
        outputs_classes = self.pred_class(outputs_classes)
        return outputs_classes.cpu().to(torch.float32), torch.cat(outputs_masks, dim=3)

    def pred_class(self, decoder_output):
        T = decoder_output.size(2)
        activation = self.activation_proj(decoder_output).softmax(dim=2)
        class_output = (decoder_output * activation).sum(dim=2, keepdim=True)
        class_output = class_output.repeat(1, 1, T, 1, 1)
        outputs_class = self.class_embed(class_output).transpose(2, 3)
        return outputs_class

    def prediction(self, outputs, mask_features):
        if self.training:
            decoder_output = self.decoder_norm(outputs)
            decoder_output = decoder_output.permute(1, 3, 0, 2, 4)
            outputs_class = self.pred_class(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)
        else:
            outputs_class, outputs_mask = self.windows_prediction(outputs, mask_features, windows=self.windows)
        return outputs_class, outputs_mask
