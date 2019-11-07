import math, copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Embedder(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embed = nn.Embedding(vocab_size, emb_dim)
    def forward(self, x):
        return self.embed(x)
    
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, emb_dim, dropout = 0.1):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.k_dim = emb_dim // heads
        self.h = heads
        
        self.q_linear = nn.Linear(emb_dim, emb_dim)
        self.v_linear = nn.Linear(emb_dim, emb_dim)
        self.k_linear = nn.Linear(emb_dim, emb_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.k_dim)
        q = self.q_linear(q).view(bs, -1, self.h, self.k_dim)
        v = self.v_linear(v).view(bs, -1, self.h, self.k_dim)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.k_dim, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.emb_dim)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(emb_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, emb_dim)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(emb_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(heads, emb_dim, dropout=dropout)
        self.norm_2 = Norm(emb_dim)
        self.ff = FeedForward(emb_dim, dropout=dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_layers, heads, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, emb_dim)
        self.pe = PositionalEncoder(emb_dim, dropout=dropout)
        self.layers = get_clones(EncoderLayer(emb_dim, heads, dropout), n_layers)
        self.norm = Norm(emb_dim)
    def forward(self, src_seq, mask):
        x = self.embed(src_seq)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(emb_dim)
        self.norm_2 = Norm(emb_dim)
        self.norm_3 = Norm(emb_dim)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, emb_dim, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, emb_dim, dropout=dropout)
        self.ff = FeedForward(emb_dim, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_layers, heads, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, emb_dim)
        self.pe = PositionalEncoder(emb_dim, dropout=dropout)
        self.layers = get_clones(DecoderLayer(emb_dim, heads, dropout), n_layers)
        self.norm = Norm(emb_dim)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, emb_dim, n_layers, heads, dropout):
        super().__init__()
        self.encoder = Encoder(in_vocab_size, emb_dim, n_layers, heads, dropout)
        self.decoder = Decoder(out_vocab_size, emb_dim, n_layers, heads, dropout)
        self.out = nn.Linear(emb_dim, out_vocab_size)
    def forward(self, src_seq, trg_seq, src_mask, trg_mask):
        e_output = self.encoder(src_seq, src_mask)
        d_output = self.decoder(trg_seq, e_output, src_mask, trg_mask)
        output = self.out(d_output)
        return output
