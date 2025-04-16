import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP
from layers.hdmixer import HDMixer  # 假设 HDMixer 的实现文件为 layers/hdmixer.py


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)

        self.gpt2 = GPT2Model.from_pretrained(configs.llm_ckp_dir)
        self.hidden_dim_of_gpt2 = 768
        self.mix = configs.mix_embeds

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化所有层时，确保它们在相同设备上
        self.hdmixer = HDMixer( configs=configs,  # 传递 configs
            q_len=self.token_len,
            d_model=self.hidden_dim_of_gpt2,
            n_heads=configs.n_heads,).to(device)
        self.norm = torch.nn.LayerNorm(768).to(device)

        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

        if configs.mlp_hidden_layers == 0:
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.token_len)
        else:
            self.encoder = MLP(self.token_len, self.hidden_dim_of_gpt2,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_hidden_activation)
            self.decoder = MLP(self.hidden_dim_of_gpt2, self.token_len,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_hidden_activation)

        self.proj_x_mark_enc = nn.Linear(8, self.hidden_dim_of_gpt2)#wind 12  exchange_rate 9  ill_ness 8


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 确保输入数据和模型在同一设备上
        x_enc = x_enc.to(device)
        #x_mark_enc = x_mark_enc.to(device)
        if x_dec is not None:
            x_dec = x_dec.to(device)
        #x_mark_dec = x_mark_dec.to(device)

        # 移动模型的部分到同一设备
        self.hdmixer = self.hdmixer.to(device)
        self.norm = self.norm.to(device)

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs, _, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)  # [bs, n_vars, seq_len] -> [bs, seq_len, n_vars]
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)  # [bs * n_vars, seq_len]

        # HDMixer 补丁处理
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]  # 计算补丁数量
        times_embeds = self.encoder(fold_out.to(torch.float32))  # [bs * n_vars, token_num, hidden_dim_of_gpt2]

        # 通过 HDMixer 处理补丁
        # Check the current shape of times_embeds
        #print("Shape of times_embeds before hdmixer:", times_embeds.shape)
        times_embeds = times_embeds.permute(0, 2, 1)
        times_embeds = self.hdmixer(times_embeds)  # 使用 HDMixer 处理
        times_embeds = times_embeds.to(device)

        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            x_mark_enc = self.proj_x_mark_enc(x_mark_enc.to(torch.float32))  # 输出形状 [256, 7, 768]

            # 由于 times_embeds 的形状是 [256, 768, 7]，需要进一步调整形状
            x_mark_enc = x_mark_enc.permute(0, 2, 1)  # 调整为 [256, 768, 7]
            times_embeds = times_embeds + self.add_scale * x_mark_enc


        #print(f"times_embeds shape: {times_embeds.shape}")
        #print(f"position_embeds shape: {self.gpt2.wpe.weight.shape}")
        times_embeds=times_embeds.permute(0, 2, 1)

        outputs = self.gpt2(inputs_embeds=times_embeds).last_hidden_state
        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)  # [bs, n_vars, token_num * token_len]
        dec_out = dec_out.permute(0, 2, 1)  # [bs, token_num * token_len, n_vars]

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

