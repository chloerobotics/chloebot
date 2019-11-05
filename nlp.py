import spacy
from torchtext import data
from Batch import MyIterator, batch_size_fn, get_len, create_masks
import re
import torch 

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