import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from model.ours.nlq_head import NLQHead


class GroundVQA(nn.Module):
    def __init__(self, lm_path, input_dim, freeze_word=False, max_v_len=256):
        super().__init__()

        if not isinstance(input_dim, int):
            input_dim = input_dim.v_dim

        self.lm: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(lm_path, local_files_only=True)

        lm_dim = self.lm.get_input_embeddings().embedding_dim
        self.lm_proj = nn.Linear(input_dim, lm_dim)
        self.v_emb = nn.Parameter(torch.randn((1, 1, lm_dim)))
        if freeze_word:
            for name, param in self.lm.named_parameters():
                if 'shared' in name:
                    param.requires_grad = False

        self.nlq_head = NLQHead(in_dim=lm_dim, max_v_len=max_v_len)

    def forward(self, v_feat, v_mask, q_token, q_mask, gt_segments, gt_labels,
                labels=None, **remains):
        # encoder
        encoder_out, mask = self.forward_encoder(v_feat, v_mask, q_token, q_mask) 
        
        # localizer
        encoder_out_v = encoder_out[:, -v_feat.shape[1]:]
        nlq_results = self.nlq_head(
            feat=encoder_out_v.permute(0, 2, 1),  # (B, D, T)
            mask=v_mask.unsqueeze(1),  # (B, 1, T)
            gt_segments=gt_segments,
            gt_labels=gt_labels
        )
        time_loss = nlq_results['final_loss'] * 1.0

        # decoder
        outputs = self.lm(
            encoder_outputs=(encoder_out,),
            attention_mask=mask,
            labels=labels,
        )
        lm_loss = outputs.loss

        total_loss = 0.5 * time_loss + 0.5 * lm_loss

        return total_loss, lm_loss, time_loss

    def generate(self, v_feat, v_mask, q_token, q_mask, v_len, **remains):
        encoder_out, mask = self.forward_encoder(v_feat, v_mask, q_token, q_mask)
        encoder_out_v = encoder_out[:, -v_feat.shape[1]:]

        nlq_results = self.nlq_head(
            feat=encoder_out_v.permute(0, 2, 1),  # (B, D, T)
            mask=v_mask.unsqueeze(1),  # (B, 1, T)
            training=False,
            v_lens=v_len
        )
        answer_tokens = self.lm.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
            attention_mask=mask,
            max_new_tokens=32
        )

        return nlq_results, answer_tokens

    def forward_encoder(self, v_feat, v_mask, q_token, q_mask):
        B, L, D = v_feat.shape
        v_feat = self.lm_proj(v_feat)
        v_feat = v_feat + self.v_emb.expand((B, L, -1))
        q_feat = self.lm.encoder.embed_tokens(q_token)
        lm_input = torch.cat([q_feat, v_feat], dim=1)
        lm_mask = torch.cat([q_mask, v_mask], dim=1)
        out = self.lm.encoder(
            inputs_embeds=lm_input,
            attention_mask=lm_mask
        )
        return out.last_hidden_state, lm_mask
