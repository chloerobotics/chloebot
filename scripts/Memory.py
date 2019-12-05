import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class RelMemCore(nn.Module):
    
    def __init__(self, mem_slots, mem_size, num_heads, dim_k=None, dropout=0.1):
        super(RelMemCore, self).__init__()
        self.mem_slots = mem_slots
        self.mem_size = mem_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.dim_k = dim_k if dim_k else self.mem_size // num_heads
        self.attn_mem_update = MultiHeadAttention(self.num_heads,self.mem_size,self.dim_k,self.dropout)
        self.normalizeMemory1 = Norm(self.mem_size)
        self.normalizeMemory2 = Norm(self.mem_size)
        self.MLP = FeedForward(self.mem_size, ff_dim=self.mem_size*2, dropout=dropout)
        self.ZGATE = nn.Linear(self.mem_size*2, self.mem_size)
        
    def initial_memory(self, , batch_size):
        """Creates the initial memory.
        TO ensure each row of the memory is initialized to be unique, 
        initialize the matrix as the identity then pad or truncate
        so that init_state is of size (mem_slots, mem_size).
        Args:
          batch size
        Returns:
          init_mem: A truncated or padded identity matrix of size (mem_slots, mem_size)
          remember_vector: (1, self.mem_size)
        """
        with torch.no_grad():
            init_mem = torch.stack([torch.eye(self.mem_slots) for _ in range(batch_size)])
            
        # Pad the matrix with zeros.
        if self.mem_size > self.mem_slots:
          difference = self.mem_size - self.mem_slots
          pad = torch.zeros((batch_size, self.mem_slots, difference))
          init_mem = torch.cat([init_mem, pad], -1)
        # Truncation. Take the first `self._mem_size` components.
        elif self.mem_size < self.mem_slots:
          init_mem = init_mem[:, :, :self.mem_size]
        
        remember_vector = torch.randn(1, 1, self.mem_size)
        remember_vector = nn.Parameter(remember_vector, requires_grad=True)
        self.register_parameter("remember_vector", remember_vector) 
        
        return init_mem, remember_vector
        
    def update_memory(self, input_vector, prev_memory):
        '''
        inputs
         input_vector (batch_size, mem_size)
         prev_memory - previous or past memory (batch_size, mem_slots, mem_size)
        output
         next_memory - updated memory (batch_size, mem_slots, mem_size)
        '''
        mem_plus_input = torch.cat([prev_memory, input_vector.unsqueeze(1)], dim=-2) 
        new_mem, scores = self.attn_mem_update(prev_memory, mem_plus_input, mem_plus_input)
        new_mem_norm = self.normalizeMemory1(new_mem + prev_memory)
        mem_mlp = self.MLP(new_mem_norm)
        new_mem_norm2 = self.normalizeMemory2(mem_mlp + new_mem_norm)
        input_stack = torch.stack([input_vector for _ in range(self.mem_slots)], dim=1)
        h_old_x = torch.cat([prev_memory, input_stack], dim = -1)
        z_t = torch.sigmoid(self.ZGATE(h_old_x)) # (batch size, memory slots, memory size)
        next_memory = (1 - z_t)*prev_memory + z_t*new_mem_norm2
        return next_memory


class MemoryTransformer(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, emb_dim, n_layers, num_heads, mem_slots, dropout):
        
        super(MemoryTransformer, self).__init__() 
        
        self.mem_slots = mem_slots
        self.mem_size = emb_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.dim_k = self.mem_size // self.num_heads
        
        self.encoder = Encoder(in_vocab_size, emb_dim, n_layers, num_heads, dropout)
        self.rmc = RelMemCore(mem_slots, mem_size=emb_dim, num_heads=num_heads)
        self.current_memory, self.rem_vec = self.rmc.initial_memory()
        self.mem_encoder = MultiHeadAttention(num_heads,self.mem_size,self.dim_k,dropout)
        self.decoder = Decoder(out_vocab_size, emb_dim, n_layers, num_heads, dropout)
        self.out = nn.Linear(emb_dim, out_vocab_size)
             
    def forward(self, src_seq, trg_seq, src_mask, trg_mask):
        e_output = self.encoder(src_seq, src_mask)
        m_output, m_scores = self.mem_encoder(e_output,self.current_memory,self.current_memory)
        d_output = self.decoder(trg_seq, m_output, src_mask, trg_mask)
        output = self.out(d_output)
        return output