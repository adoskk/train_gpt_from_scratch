import torch.nn as nn
import torch

def softmax_attention(q, k, v):
    # q: (b, h, seq_len, head_dim)
    # k: (b, h, seq_len, head_dim)
    # v: (b, h, seq_len, head_dim)
    
    score = torch.einsum("ijkl,ijlm->ijkm", q, k)
    score = nn.Softmax(dim=-1)(score/torch.sqrt(q.size()[-1]))
    attention = torch.einsum("ijkm,ijml->ikl", score, v)
    return attention, score
  

class DecoderLayer(nn.module):
    def __init__(self, d_model):
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        
    def forward(self, x, head_dim):
        # x: (b, seq_len, d_model)
        b, seq_len, d_model = x.size()
        
        q, k, v = self.WQ(x), self.WK(x), self.WV(x)
        
        q = q.view(b, seq_len, int(d_model/head_dim), head_dim)
        k = k.view(b, seq_len, int(d_model/head_dim), head_dim)
        v = v.view(b, seq_len, int(d_model/head_dim), head_dim)
        
        attention, _ = softmax_attention(q, k, v)
        
        return attention
        
        
class GPTMini(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=512):
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        
        s_ind, i_ind = torch.arange(seq_len), 1/10000**(2*torch.arange(d_model)/d_model)
        self.pos = torch.sin(torch.einsum("i,j->ij", s_ind, i_ind))
        self.decoder = DecoderLayer()
        
    def forward(self, x):
        
        x = self.embedding_layer(x) + self.pos
        x = self.decoder(x)
        
        return x
        
            
    
        
        