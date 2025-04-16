import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP
from layers.danet import DANetHead


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)

        self.gpt2 = GPT2Model.from_pretrained(configs.llm_ckp_dir)
        self.hidden_dim_of_gpt2 = 768
        self.mix = configs.mix_embeds

        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.token_len)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.token_len, self.hidden_dim_of_gpt2,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_gpt2, self.token_len,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_activation)
        self.proj_x_mark_enc = nn.Linear(12, self.hidden_dim_of_gpt2)  # wind:12 exchange_rate 9  illness 8

        # Initialize dual attention network
        self.danet = DANetHead(self.hidden_dim_of_gpt2, self.hidden_dim_of_gpt2, norm_layer=nn.BatchNorm2d)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs, _, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]
        times_embeds = self.encoder(fold_out.to(torch.float32))
        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            x_mark_enc = self.proj_x_mark_enc(x_mark_enc.to(torch.float32))
            times_embeds = times_embeds + self.add_scale * x_mark_enc

        outputs = self.gpt2(inputs_embeds=times_embeds).last_hidden_state

        # Apply the dual attention network
        outputs = outputs.view(bs, n_vars, token_num, -1).permute(0, 3, 1, 2)  # reshape to (N, C, H, W) format
        attn_out, sa_output, sc_output = self.danet(outputs)
        attn_out = attn_out.permute(0, 2, 3, 1).reshape(bs * n_vars, token_num, -1)  # reshape back

        dec_out = self.decoder(attn_out)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
