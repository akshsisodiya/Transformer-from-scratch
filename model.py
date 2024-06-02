import torch
from torch import nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pos_enc = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        # unsqueeze for batch (1, seq_len, d_model)
        pos_enc = pos_enc.unsqueeze(0)

        # save in buffee (todo)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        x = x + (self.pos_enc[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.linear2(x)
        
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dropout:float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]

        attention_score = (query @ key.transpose(-2,-1) / math.sqrt(d_k))
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)

        attention_score = attention_score.softmax(dim=-1)

        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shpae[1], self.n_heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1,2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, self.dropout)
        x = x.transpose(1,2).contigous().view(x.shape[0], -1, self.n_heads*self.d_k)

        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout:float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return 