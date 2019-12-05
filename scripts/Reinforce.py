import re, math, time
import numpy as np

from nltk.corpus import wordnet

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def reinforce_chloe(input_str, chloe, opt, infield, outfield):

    input_sequence = string2tensor(input_str, infield)
    input_mask = (input_sequence != infield.vocab.stoi['<pad>']).unsqueeze(-2)
    chloe.eval()
    encoding = chloe.encoder(input_sequence, input_mask)
    init_tok = outfield.vocab.stoi['<sos>'] 
    decoder_input = torch.LongTensor([[init_tok]])
    logprobs = torch.Tensor([[]])
    for pos in range(opt.max_len):
        decoder_input_mask = nopeak_mask(size=pos+1, opt=opt)
        out = chloe.out(chloe.decoder(decoder_input, encoding, input_mask, decoder_input_mask))
        softout = F.softmax(out, dim=-1)
        distr = Categorical(probs=softout)
        action = distr.sample()[:,-1].unsqueeze(0)
        logprob = -distr.log_prob(action)[:,-1].unsqueeze(0)
        decoder_input = torch.cat((decoder_input, action), dim=1)
        logprobs = torch.cat((logprobs, logprob), dim=1)
        if outfield.vocab.itos[action] == '<eos>':
            de_str = ' '.join([outfield.vocab.itos[tok] for tok in decoder_input[0]])
            return decoder_input, de_str, logprobs
        
    de_str = ' '.join([outfield.vocab.itos[tok] for tok in decoder_input[0]])
    return decoder_input, de_str, logprobs