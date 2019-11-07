import re, math
import numpy as np

from nltk.corpus import wordnet

import torch
from torch.autograd import Variable
import torch.nn.functional as F

def nopeak_mask(size, opt):
    "Mask out subsequent positions. aka subsequent_mask"
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  torch.from_numpy(np_mask) == 0
    if opt.device != -1:
      np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg, opt):
    
    src_mask = (src != opt.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask

def talk_to_model(sentence, model, opt, SRC, TRG):
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))
    if opt.device != -1:
        sentence = sentence.cuda()
    sentence = beam_search(sentence, model, SRC, TRG, opt)
    return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)

def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]
    return 0

def beam_search(src, model, SRC, TRG, opt):
    #print(next(model.parameters()).is_cuda,src.is_cuda)
    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)
    #e_outputs.shape [batch_size, seq_len, emd_dim] print(outputs, e_outputs.shape, log_scores)
    eos_tok = TRG.vocab.stoi['<eos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    ind = None
    for i in range(2, opt.max_len):
        trg_mask = nopeak_mask(i, opt)
        out = model.out(model.decoder(outputs[:,:i], e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k)
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol
        num_finished_sentences = len([s for s in sentence_lengths if s > 0])
        if num_finished_sentences == opt.k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    if ind is None:
        print(outputs[0]==eos_tok)
        print((outputs[0]==eos_tok).nonzero())
        length = (outputs[0]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
    else:
        length = (outputs[ind]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])

def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    row = k_ix // k
    col = k_ix % k
    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]
    log_scores = k_probs.unsqueeze(0)
    return outputs, log_scores

def init_vars(src, model, SRC, TRG, opt):
    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    #print(next(model.parameters()).is_cuda,src.is_cuda,src_mask.is_cuda)
    e_output = model.encoder(src, src_mask)
    outputs = torch.LongTensor([[init_tok]])
    if opt.device != -1:
        outputs = outputs.cuda()
    trg_mask = nopeak_mask(1, opt)
    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)
    #print(out.shape)
    probs, ix = out[:, -1].data.topk(opt.k)
    #print('probs, ix',probs, ix)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    #print(log_scores)
    outputs = torch.zeros(opt.k, opt.max_len).long()
    if opt.device != -1:
        outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    e_outputs = torch.zeros(opt.k, e_output.size(-2),e_output.size(-1))
    if opt.device != -1:
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = e_output[0]
    return outputs, e_outputs, log_scores

def multiple_replace(dict, text):
  # Create a regular expression from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)

def greedy_talk(sentence, model, opt, SRC, TRG, max_len):
    model.eval()
    input_seq = []
    sentence = SRC.preprocess(sentence)
    #print(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0:
            input_seq.append(SRC.vocab.stoi[tok])
        else:
            input_seq.append(get_synonym(tok, SRC))
    input_seq = torch.LongTensor([input_seq])
    if opt.device != -1:
        input_seq = input_seq.cuda()
    #print(input_seq)
    encoding_mask = (input_seq != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    #print(encoding_mask)
    encoding = model.encoder(input_seq, encoding_mask)
    #print(encoding.shape)
    start_index = outfield.vocab.stoi['<sos>']
    ys = torch.ones(1, 1).fill_(start_index).type_as(input_seq.data)
    #print(ys)
    for i in range(max_len-1):
        np_mask = nopeak_mask(ys.size(1), opt).type_as(input_seq.data)
        print(encoding.shape, encoding_mask.shape, ys.shape, np_mask.shape)
        out = model.decoder(encoding, encoding_mask, ys, np_mask)
        break 
    #print('sentence.is_cuda',sentence.is_cuda)
    #sentence = beam_search(sentence, model, SRC, TRG, opt)
    #sentence = multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)
    return sentence