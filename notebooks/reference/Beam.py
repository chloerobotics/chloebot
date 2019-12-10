import torch
import torch.nn.functional as F
import math
import numpy as np 

import re

def num_batches(train):

    for i, b in enumerate(train):
        pass
    
    return i + 1

def nopeak_mask(size, opt):
    "Mask out subsequent positions. aka subsequent_mask"
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  torch.from_numpy(np_mask) == 0
    if opt.device == torch.device("cuda:0") and next(model.parameters()).is_cuda:
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

def string2tensor(string, inputfield, explain=False):
    '''
    input:
        string (str) input sentence
        inputfield a PyTorch torchtext.data.Field object
        explain, set this to True if you want to see how the sentence was split 
    output:
        sequence of tokens (torch tensor of integers) shape  
    '''
    sentence = inputfield.preprocess(string)
    if explain: print(sentence)
    integer_sequence = []
    for tok in sentence:
        if inputfield.vocab.stoi[tok] != 0:
            integer_sequence.append(inputfield.vocab.stoi[tok])
        else:
            integer_sequence.append(get_synonym(tok, inputfield))
    return torch.LongTensor([integer_sequence])

def get_synonym(word, field, explain=False):
    syns = wordnet.synsets(word)
    for s in syns:
        if explain: print('synonym:', s.name())
        for l in s.lemmas():
            if explain: print('-lemma:', l.name())
            if field.vocab.stoi[l.name()] != 0:
                if explain: print('found in vocab', l.name())
                return field.vocab.stoi[l.name()]
    return 0 # if we cannot find a synonym, return 0

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

def translate_sentence(sentence, model, opt, SRC, TRG):
    
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = torch.LongTensor([indexed])
    
    sentence = beam_search(sentence, model, SRC, TRG, opt)

    return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)

def translate(opt, model, SRC, TRG):
    sentences = opt.text.lower().split('.')
    translated = []

    for sentence in sentences:
        translated.append(translate_sentence(sentence + '.', model, opt, SRC, TRG).capitalize())

    return (' '.join(translated))

def init_vars(src, model, SRC, TRG, opt):
    
    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    e_output = model.encoder(src, src_mask)
    e_output, src_mask = model.concat_mem(e_output, src_mask)
    
    outputs = torch.LongTensor([[init_tok]])
    trg_mask = nopeak_mask(1, opt)

    #print(outputs.shape, trg_mask.shape, e_output.shape, src_mask.shape)

    if outputs.size(0) < e_output.size(0):
            outputs = outputs.repeat(e_output.size(0),1)
            trg_mask = trg_mask.repeat(e_output.size(0),1,1)

    
    #print(outputs.shape, trg_mask.shape, e_output.shape, src_mask.shape)
    
    out = model.out(model.decoder(outputs, trg_mask, e_output, src_mask))

    out = F.softmax(out, dim=-1)
    
    probs, ix = out[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.k, opt.max_len).long()

    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    
    e_outputs = torch.zeros(opt.k, e_output.size(-2),e_output.size(-1))

    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, src_mask, log_scores

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

def beam_search(src, model, SRC, TRG, opt):
    

    outputs, e_outputs, src_mask, log_scores = init_vars(src, model, SRC, TRG, opt)
    eos_tok = TRG.vocab.stoi['<eos>']
    #src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    ind = None
    for i in range(2, opt.max_len):
    
        trg_mask = nopeak_mask(i, opt)

        #print(outputs[:,:i].shape, trg_mask.shape, e_outputs.shape, src_mask.shape)

        model.d_output = model.decoder(outputs[:,:i], trg_mask, e_outputs, src_mask)

        out = model.out(model.d_output)

        out = F.softmax(out, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k)
        
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long)
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
        length = (outputs[0]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
    
    else:
        length = (outputs[ind]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])