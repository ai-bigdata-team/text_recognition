import torch 
import torch.nn as nn
import torch.nn.functional as F 
import math

from encoder import TransformerBlock
from pos_embeddings import PositionalEncoding, LearnedPositionalEncoding
class EncoderOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed  = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config)
                                    for _ in range(config.n_decoder_layers)] )
        self.norm = nn.LayerNorm(config.d_model)
    def forward(self, x, mask = None):
        for layer in self.layers:
            x  = layer(x, mask = mask)
        x = self.norm(x)
        return x
    def get_attn_maps(self, x, mask = None):
        attn_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attn_maps.append(attn_map)
            x = layer(x, mask = mask)
        return attn_maps

class DecoderOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) 
                                    for _ in range(config.n_encoder_layers)] )
        mask = torch.triu(torch.ones(self.config.seq_length, self.config.seq_length), diagonal=1)
        self.register_buffer('mask', mask == 0)
        self.fc = nn.Linear(config.d_model, config.vocab_size)
    def forward(self, x):
        for layer in self.layers:
            x  = layer(x, mask = self.mask)
        x = self.fc(x)
        return x
    def get_attn_maps(self, x, mask = None):
        attn_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attn_maps.append(attn_map)
            x = layer(x, mask = self.mask)
        return attn_maps


class LanguageTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        max_seq_length,
        pos_dropout,
        trans_dropout,
    ):
        super().__init__()

        self.d_model = d_model
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        # self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)
        self.learned_pos_enc = LearnedPositionalEncoding(d_model, pos_dropout, max_seq_length)

        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            trans_dropout,
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Shape:
            - src: (W, N, C)
            - tgt: (T, N)
            - src_key_padding_mask: (N, S)
            - tgt_key_padding_mask: (N, T)
            - memory_key_padding_mask: (N, S)
            - output: (N, T, E)

        """
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(src.device)

        src = self.pos_enc(src * math.sqrt(self.d_model))
        #        src = self.learned_pos_enc(src*math.sqrt(self.d_model))

        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

        output = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask.float(),
            memory_key_padding_mask=memory_key_padding_mask,
        )
        #        output = rearrange(output, 't n e -> n t e')
        output = output.transpose(0, 1)
        return self.fc(output)

    def gen_nopeek_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )

        return mask

    def forward_encoder(self, src):
        src = self.pos_enc(src * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src)
        return memory

    def forward_decoder(self, tgt, memory):
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(tgt.device)
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        #        output = rearrange(output, 't n e -> n t e')
        output = output.transpose(0, 1)

        return self.fc(output), memory

    def expand_memory(self, memory, beam_size):
        memory = memory.repeat(1, beam_size, 1)
        return memory

    def get_memory(self, memory, i):
        memory = memory[:, [i], :]
        return memory

