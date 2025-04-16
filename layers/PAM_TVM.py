import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .hierarchical_mm_tvm import graph_mm as graph_mm_tvm


class PyramidalAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, dropout, normalize_before, q_k_mask, k_q_mask):
        super(PyramidalAttention, self).__init__()
        self.normalize_before = normalize_before
        self.n_heads = n_heads
        self.d_k = d_k

        #d_model=768,d_k=128,n_heads=4
        self.w_qs = nn.Linear(7, n_heads * d_k, bias=False)
        #self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=False)
        #self.w_vs = nn.Linear(d_model, n_heads * d_k, bias=False)

        self.w_ks = nn.Linear(7, n_heads * d_k, bias=False)
        self.w_vs = nn.Linear(7, n_heads * d_k, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        #self.fc = nn.Linear(d_k * n_heads, d_model)
        self.fc = nn.Linear(d_k * n_heads, 7)
        nn.init.xavier_uniform_(self.fc.weight)

        self.layer_norm = nn.LayerNorm(7, eps=1e-6)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_fc = nn.Dropout(dropout)
        self.q_k_mask = q_k_mask
        self.k_q_mask = k_q_mask

    def forward(self, hidden_states):
        residual = hidden_states #[256,768,7]
        hidden_states = hidden_states
        bsz, seq_len,  _ = hidden_states.size()

        self.linear_layer = nn.Linear(7, seq_len)
        device=torch.cuda.set_device(0)
        self.linear_layer = self.linear_layer.to(device)

        q = hidden_states  #[256,768,7]
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q) ##[256,1024]
        k = self.w_ks(hidden_states) #[256, 768,1024]
        v = self.w_vs(hidden_states) #[256, 768,1024]
        q /= math.sqrt(self.d_k)

        q = q.view(bsz, seq_len, self.n_heads, self.d_k)
        k = k.view(bsz, seq_len, self.n_heads, self.d_k)
        q = q.float().contiguous()
        k = k.float().contiguous()
        # attn_weights.size(): (batch_size, L, num_heads, 11)

       # q shape: torch.Size([256, 768, 4, 128]), device: cuda:0
       # k shape: torch.Size([256, 768, 4, 128]), device: cuda:0
       # q_k_mask shape: torch.Size([768, 64]), device: cuda:0
       # k_q_mask shape: torch.Size([768, 64]), device: cuda:0

        attn_weights = graph_mm_tvm(q, k, self.q_k_mask, self.k_q_mask, False, -1000000000)
       # print("attn_weights.shape before dropout:", attn_weights.shape)   #[256, 768, 4, 64]
        #attn_weights = self.dropout_attn(F.softmax(attn_weights, dim=-1))  #softmax() 激活函数
        attn_weights = F.softmax(attn_weights, dim=-1)

        v = v.view(bsz, seq_len, self.n_heads, self.d_k)
        v = v.float().contiguous()
        # is_t1_diagonaled=True

        # print("attn_weights.shape:", attn_weights.shape)
        # print("v.shape:", v.shape)

        attn = graph_mm_tvm(attn_weights, v, self.q_k_mask, self.k_q_mask, True, 0)
        attn = attn.reshape(bsz, seq_len, self.n_heads * self.d_k).contiguous()  #[256,768,1024]
        #context = self.dropout_fc(self.fc(attn))
        context = self.fc(attn)

        context += residual  #context=Tensor:(256,768,7)  residual=Tensor:(256,768,7)

        if not self.normalize_before:
            context = self.layer_norm(context)

        return context

