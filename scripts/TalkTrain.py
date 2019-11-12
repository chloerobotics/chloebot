import re, math, time
import numpy as np

from nltk.corpus import wordnet

import torch
from torch.autograd import Variable
import torch.nn.functional as F

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

def talk_to_model(sentence, model, opt, SRC, TRG, explain=False):
    try:
        model.eval()
        indexed = []
        sentence = SRC.preprocess(sentence)
        for tok in sentence:
            if SRC.vocab.stoi[tok] != 0:
                indexed.append(SRC.vocab.stoi[tok])
            else:
                indexed.append(get_synonym(tok, SRC))
        sentence = Variable(torch.LongTensor([indexed]))

        if opt.device == torch.device("cuda:0") and next(model.parameters()).is_cuda:
            sentence = sentence.cuda()

        sentence = beam_search(sentence, model, SRC, TRG, opt)
        return multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)
    except Exception as e:
        if explain: print(e)
        return "haha, got me, you will have to teach me how to reply to that, I haven't learned yet"

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
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long)

        if opt.device == torch.device("cuda:0") and next(model.parameters()).is_cuda:
            sentence_lengths = sentence_lengths.cuda()

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
        #print(outputs[0]==eos_tok)
        #print((outputs[0]==eos_tok).nonzero())
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
    if opt.device == torch.device("cuda:0") and next(model.parameters()).is_cuda:
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
    if opt.device == torch.device("cuda:0") and next(model.parameters()).is_cuda:
        outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    e_outputs = torch.zeros(opt.k, e_output.size(-2),e_output.size(-1))
    if opt.device == torch.device("cuda:0") and next(model.parameters()).is_cuda:
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = e_output[0]
    return outputs, e_outputs, log_scores

def multiple_replace(dict, text):
  # Create a regular expression from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)

def trainer(model, data_iterator, options, optimizer, scheduler):

    if torch.cuda.is_available() and options.device == torch.device("cuda:0"):
        print("a GPU was detected, model will be trained on GPU")
        model = model.cuda()
    else:
        print("training on cpu")

    model.train()
    start = time.time()
    best_loss = 100
    for epoch in range(options.epochs):
        total_loss = 0
        for i, batch in enumerate(data_iterator): 
            src = batch.listen.transpose(0,1)
            trg = batch.reply.transpose(0,1)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, options)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            optimizer.zero_grad()
            batch_loss = F.cross_entropy(preds.view(-1, preds.size(-1)), 
                                         ys, ignore_index = options.trg_pad)
            batch_loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += batch_loss.item()

        epoch_loss = total_loss/(num_batches(data_iterator)+1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), options.save_path)
        print("%dm: epoch %d loss = %.3f" %((time.time() - start)//60, epoch, epoch_loss))
        total_loss = 0

    return model

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart = 0
        self._cycle_counter = 0
        self._cycle_factor = 1.0
        self._updated_cycle_len = T_max
        self._initialized = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs