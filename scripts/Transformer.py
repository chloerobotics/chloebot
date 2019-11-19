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

    def __init__(self, emb_dim, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self. emb_dim =  emb_dim
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on pos and i
        # this matrix of shape (1, input_seq_len, emb_dim) is
        # cut down to shape (1, input_seq_len, emb_dim) int he forward pass
        # to be broadcasted across each sampel in the batch 
        pe = torch.zeros(max_seq_len, emb_dim)
        for pos in range(max_seq_len):
            for i in range(0, emb_dim, 2):
                wavelength = 10000 ** ((2 * i)/ emb_dim)
                pe[pos, i] = math.sin(pos / wavelength)
                pe[pos, i + 1] = math.cos(pos / wavelength)
        pe = pe.unsqueeze(0) # add a batch dimention to your pe matrix 
        self.register_buffer('pe', pe)#block of data(persistent buffer)->temporary memory
 
    def forward(self, x):
        '''
        input: sequence of vectors  
               shape (batch size, input sequence length, vector dimentions)
        output: sequence of vectors of same shape as input with positional
                aka time encoding added to each sample 
                shape (batch size, input sequence length, vector dimentions)
        '''
        x = x * math.sqrt(self. emb_dim) # make embeddings relatively larger
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe #add constant to embedding
        return self.dropout(x)
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Norm(nn.Module):
    def __init__(self, emb_dim, eps = 1e-6):
        super().__init__()
        self.size = emb_dim
        # alpha and bias are learnable parameters that scale and shift
        # the representations respectively, aka stretch and translate 
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps #prevents divide by zero explosions 
    
    def forward(self, x):
        '''
        input/output: x/norm shape (batch size, sequence length, embedding dimensions)
        '''
        norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps)
        norm = self.alpha * norm + self.bias
        return norm

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_dim, dim_k = None, dropout = 0.1):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.dim_k = dim_k if dim_k else emb_dim // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(emb_dim, self.dim_k*num_heads)
        self.v_linear = nn.Linear(emb_dim, self.dim_k*num_heads)
        self.k_linear = nn.Linear(emb_dim, self.dim_k*num_heads)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.dim_k*num_heads,emb_dim)
    
    def attention(self, q, k, v, dim_k, mask=None, dropout=None, explain=False):
        k = k.transpose(-2, -1)
        if explain: print('q, k', q.shape, k.shape)
        # matrix multiplication is done using the last two dimensions
        # (batch_size,num_heads,seq_len,dim_k)X(batch_size,num_heads,dim_k,seq_len)
        scores = torch.matmul(q, k) / math.sqrt(dim_k) # = (batch_size,num_heads,seq_len,seq_len)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if explain: print('scores', scores.shape)
        if dropout is not None: scores = dropout(scores)
        # (batch_size,num_heads,seq_len,seq_len)X(batch_size,num_heads,seq_len,dim_k)
        output = torch.matmul(scores, v)
        return output # = (batch_size,num_heads,seq_len,dim_k)
    
    def forward(self, q, k, v, mask=None, explain=False):
        '''
        inputs:
            q,k,v are shape (batch size, sequence length, embedding dimensions)
            source_mask of shape (batch size, 1, sequence length)
        outputs: sequence of vectors, re-represented using attention
            shape (batch size, sequence length, embedding dimensions)
        use:
            the encoder layer places the same source vector sequence into q,k,v 
            and source_mask into mask
            the decoder layer uses this twice, once with decoder inputs as q,k,v 
            and target mask as mask. then with decoder inputs as q, encoder outputs
            as k, v and source mask as mask
        '''
        # k,q,v are each shape (batch size, sequence length, dim_k * num_heads)
        batch_size = q.size(0)
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        if explain: print("(batch size, sequence length, dim_k * num_heads)", k.shape)
        # k,q,v are each shape (batch size, sequence length, num_heads, dim_k)
        k = k.view(batch_size,-1,self.num_heads,self.dim_k)
        q = q.view(batch_size,-1,self.num_heads,self.dim_k)
        v = v.view(batch_size,-1,self.num_heads,self.dim_k)
        # transpose to shape (batch_size, num_heads, sequence length, dim_k)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        if explain: print("(batch_size,num_heads,seq_length,dim_k)",k.shape)
        # calculate attention using function we will define next
        attn = self.attention(q, k, v, self.dim_k, mask, self.dropout, explain)
        if explain: print("attn(batch_size,num_heads,seq_length,dim_k)", attn.shape)
        # concatenate heads and 
        concat=attn.transpose(1,2).contiguous().view(batch_size,-1,self.dim_k*self.num_heads)
        if explain: print("concat.shape", concat.shape)
        # put through final linear layer
        output = self.out(concat)
        if explain: print("output", output.shape)
        return output

class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(emb_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, emb_dim)
    
    def forward(self, x):
        x = self.dropout(F.leaky_relu(self.linear_1(x)))
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
    def forward(self, source_sequence, source_mask):
        '''
        input:
        source_sequence (sequence of source tokens) of shape (batch size, sequence length)
        source_mask (mask over input sequence) of shape (batch size, 1, sequence length)
        output: x.shape after layers and after norm both are of shape
        (batch size, sequence length, embedding dimensions)
        '''
        x = self.embed(source_sequence)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, source_mask)
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

