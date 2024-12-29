import torch.nn as nn
import torch

def softmax_attention(q, k, v, mask_bool):
    # q: (b, h, seq_len, head_dim)
    # k: (b, h, seq_len, head_dim)
    # v: (b, h, seq_len, head_dim)
    
    score = torch.einsum("ijkl,ijlm->ijkm", q, k)
    score = nn.Softmax(dim=-1)(score/torch.sqrt(q.size()[-1]))
    score.masked_fill_(mask_bool, -torch.inf)
    attention = torch.einsum("ijkm,ijml->ikl", score, v)
    return attention, score
  

class DecoderLayer(nn.module):
    def __init__(self, seq_len, d_model):
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1))
        
        self.mlp = nn.Sequential([nn.Linear(d_model, 4*d_model),
                                  nn.GELU(),
                                  nn.Linear(4*d_model, d_model)])
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, head_dim):
        # x: (b, token_num, d_model)
        b, token_num, d_model = x.size()
        
        q, k, v = self.WQ(x), self.WK(x), self.WV(x)
        
        q = q.view(b, token_num, int(d_model/head_dim), head_dim)
        k = k.view(b, token_num, int(d_model/head_dim), head_dim)
        v = v.view(b, token_num, int(d_model/head_dim), head_dim)
        
        mask_bool = self.mask.bool()[:token_num, :token_num]
        
        attention, _ = softmax_attention(q, k, v, mask_bool)
        
        attention = attention.contiguous().view(b, token_num, d_model)
        
        output = self.layer_norm(x + attention)
        output = self.layer_norm(self.mlp(output) + output)
        
        return output
        
        
def positional_encoding(shape, type):
    seq_len, d_model = shape
    if type == "sinusoidal":
        pos = torch.zeros((seq_len, d_model))
        s_ind, i_ind = torch.arange(seq_len), 1/10000**(2*torch.arange(d_model)/d_model)
        pos[:, ::2] = torch.sin(torch.einsum("i,j->ij", s_ind, i_ind))
        pos[:, 1::2] = torch.cos(torch.einsum("i,j->ij", s_ind, i_ind))
    
    return pos
        
class GPTMini(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=512):
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        
        self.pos = positional_encoding((seq_len, d_model), "sinusoidal")
        
        self.decoder = nn.Sequential([DecoderLayer(seq_len, d_model) for _ in range(10)])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        
        output = self.embedding_layer(x) + self.pos
        output = self.decoder(output)
        output = self.layer_norm(output)
        output = self.linear(output)

        return output
        
            
    
        
        