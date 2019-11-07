import re
import numpy as np
import spacy

import torch
from torchtext import data
from torch.autograd import Variable

from torchtext import data

class MyIterator(data.Iterator):
    '''
    patch on Torchtext's batching process that makes it more efficient
    http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks
    '''
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class tokenize(object):
    
    def __init__(self, lang):
        self.nlp = spacy.load(lang)
            
    def tokenizer(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]

def csv2datatools(path, lang, options = None):
    if options == None:
        options = options()
        options.batchsize = 4
    options.device = torch.device("cuda:0") if torch.cuda.is_available() else -1
    language_class = tokenize(lang)
    input_field = data.Field(lower=True, tokenize=language_class.tokenizer,
                            unk_token='<unk>', init_token='<sos>', eos_token='<eos>')
    output_field = data.Field(lower=True, tokenize=language_class.tokenizer, 
                            unk_token='<unk>', init_token='<sos>', eos_token='<eos>')

    data_fields = [('input_text', input_field), ('output_text', output_field)]
    trainingset = data.TabularDataset(path,skip_header=True,format='csv', 
                                     fields=data_fields)
    input_field.build_vocab(trainingset)
    output_field.build_vocab(trainingset)
    training_iterator = MyIterator(trainingset, batch_size=options.batchsize, 
                        device=options.device, repeat=False, 
                        sort_key=lambda x: (len(x.input_text), len(x.output_text)), 
                        train=True, shuffle=True)
    options.src_pad = input_field.vocab.stoi['<pad>']
    options.trg_pad = output_field.vocab.stoi['<pad>']
    return training_iterator, input_field, output_field, options

class Options:
    def __init__(self, batchsize=4):
        self.batchsize = batchsize
