import torch
from torch import nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        #print("hidden_dim: ", hidden_dim)  #1024
        #hidden_dim = multiple_of * ((2 * hidden_dim // 3 + multiple_of - 1) // multiple_of)
        #print("dim: ", dim)  #96
        #print("hidden_dim after: ", hidden_dim)  #688
        self.w1 = nn.Linear(hidden_dim,dim)
        self.w2 = nn.Linear(dim,hidden_dim)
        self.w3 = nn.Linear(hidden_dim,dim)
        self.dropout = nn.Dropout(dropout)
        '''self.w1:  Linear(in_features=96, out_features=688, bias=True)
           self.w2:  Linear(in_features=688, out_features=96, bias=True)
           self.w3:  Linear(in_features=96, out_features=688, bias=True)'''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print("x.shape:",x.shape)  #x.shape: torch.Size([256, 7, 1024])
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))



class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=256, 
                 hidden_layers=2, 
                 dropout=0.1,
                 activation='tanh'): 
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swiglu':
            self.activation = SwiGLU(f_in, hidden_dim, 8, dropout)
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim), 
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]
        
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y
