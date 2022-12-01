"""
Language Modeling with nn.Transformer and TorchText
===============================================================
This is a tutorial on training a sequence-to-sequence model that uses the
`nn.Transformer <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`__ module.
The PyTorch 1.2 release includes a standard transformer module based on the
paper `Attention is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`__.
Compared to Recurrent Neural Networks (RNNs), the transformer model has proven
to be superior in quality for many sequence-to-sequence tasks while being more
parallelizable. The ``nn.Transformer`` module relies entirely on an attention
mechanism (implemented as
`nn.MultiheadAttention <https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>`__)
to draw global dependencies between input and output. The ``nn.Transformer``
module is highly modularized such that a single component (e.g.,
`nn.TransformerEncoder <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html>`__)
can be easily adapted/composed.
.. image:: ../_static/img/transformer_architecture.jpg
"""

######################################################################
# Define the model
# ----------------
#


######################################################################
# In this tutorial, we train a ``nn.TransformerEncoder`` model on a
# language modeling task. The language modeling task is to assign a
# probability for the likelihood of a given word (or a sequence of words)
# to follow a sequence of words. A sequence of tokens are passed to the embedding
# layer first, followed by a positional encoding layer to account for the order
# of the word (see the next paragraph for more details). The
# ``nn.TransformerEncoder`` consists of multiple layers of
# `nn.TransformerEncoderLayer <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html>`__.
# Along with the input sequence, a square attention mask is required because the
# self-attention layers in ``nn.TransformerEncoder`` are only allowed to attend
# the earlier positions in the sequence. For the language modeling task, any
# tokens on the future positions should be masked. To produce a probability
# distribution over output words, the output of the ``nn.TransformerEncoder``
# model is passed through a linear layer followed by a log-softmax function.
#

import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg')
import numpy as np
import os
import datetime

assert torch.cuda.is_available(), 'GPU not available'
device = torch.device('cuda')

class TransformerModel(nn.Module):

  def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
               nlayers: int, dropout: float = 0.5):
    super().__init__()
    self.model_type = 'Transformer'

    # frequency-based representation of word position with dropout
    self.pos_encoder = PositionalEncoding(d_model,dropout)

    # create self-attention + feedforward network module
    # d_model: number of input features
    # nhead: number of heads in the multiheadattention models
    # dhid: dimension of the feedforward network model
    # dropout: dropout value
    encoder_layers = TransformerEncoderLayer(d_model,nhead,d_hid,dropout)

    # stack of nlayers self-attention + feedforward layers
    # nlayers: number of sub-encoder layers in the encoder
    self.transformer_encoder = TransformerEncoder(encoder_layers,nlayers)

    # lookup table to store word embeddings and retrieve them using indices
    # ntoken: size of the dictionary of embeddings
    # d_model: the size of the embedding vector
    self.encoder = nn.Embedding(ntoken,d_model)

    # decoder for creating tokens from embeddings
    self.decoder = nn.Linear(d_model,ntoken)

    # store hyperparameters
    self.d_model = d_model

    self.init_weights()

  def init_weights(self) -> None:
    initrange = 0.1
    # encoder and decoder are initialized with uniform random numbers between -.1 and .1
    self.encoder.weight.data.uniform_(-initrange,initrange)
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform_(-initrange,initrange)

  def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
    """
    Args:
      src: Tensor, shape [seq_len,batch_size]
      src_mask: Tensor, shape [seq_len,seq_len]
    Returns:
      output Tensor of shape [seq_len, batch_size, ntoken]
    """

    # embed word tokens into d_model space, multiple by sqrt(d_model) for reasons?
    src = self.encoder(src) * math.sqrt(self.d_model)

    # add in the positional encoding of where in the sentence the words occur
    # it is weird to me that these are added, but I guess it would be almost
    # the same to have these be combined in a single linear layer
    src = self.pos_encoder(src)

    # main transformer layers
    output = self.transformer_encoder(src,src_mask)

    # back to tokens
    output = self.decoder(output)

    return output

def generate_square_subsequent_mask(sz: int) -> Tensor:
  """
  Generates an upper-triangular matrix of -inf, with zeros on and below the diagonal.
  This is used to restrict attention to the past when predicting future words.
  """
  return torch.triu(torch.ones(sz,sz) * float('-inf'), diagonal=1)

######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()

    # during training, randomly zero some of the inputs with probability p=dropout
    self.dropout = nn.Dropout(p=dropout)

    # compute sine and cosine waves at different frequencies
    # pe[:,0,i] will have a different value for each word (or whatever)
    # will be sines for even i, cosines for odd i,
    # exponentially decreasing frequencies with i
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
    pe = torch.zeros(max_len,1,d_model)
    pe[:,0,0::2] = torch.sin(position * div_term)
    pe[:,0,1::2] = torch.cos(position * div_term)

    # buffers will be saved with model parameters, but are not model parameters
    self.register_buffer('pe', pe)

  def forward(self, x: Tensor) -> Tensor:
    """
    Args:
      x: Tensor, shape [seq_len, batch_size, embedding_dim]
    """

    # add positional encoding
    x = x + self.pe[:x.size(0)]

    # zero out a randomly selected subset of entries
    return self.dropout(x)

######################################################################
# Load and batch data
# -------------------
#


######################################################################
# This tutorial uses ``torchtext`` to generate Wikitext-2 dataset.
# To access torchtext datasets, please install torchdata following instructions at
# https://github.com/pytorch/data.
# %%
#  .. code-block:: bash
#
#      %%bash
#      pip install torchdata
#
# The vocab object is built based on the train dataset and is used to numericalize
# tokens into tensors. Wikitext-2 represents rare tokens as `<unk>`.
#
# Given a 1-D vector of sequential data, ``batchify()`` arranges the data
# into ``batch_size`` columns. If the data does not divide evenly into
# ``batch_size`` columns, then the data is trimmed to fit. For instance, with
# the alphabet as the data (total length of 26) and ``batch_size=4``, we would
# divide the alphabet into 4 sequences of length 6:
#
# .. math::
#   \begin{bmatrix}
#   \text{A} & \text{B} & \text{C} & \ldots & \text{X} & \text{Y} & \text{Z}
#   \end{bmatrix}
#   \Rightarrow
#   \begin{bmatrix}
#   \begin{bmatrix}\text{A} \\ \text{B} \\ \text{C} \\ \text{D} \\ \text{E} \\ \text{F}\end{bmatrix} &
#   \begin{bmatrix}\text{G} \\ \text{H} \\ \text{I} \\ \text{J} \\ \text{K} \\ \text{L}\end{bmatrix} &
#   \begin{bmatrix}\text{M} \\ \text{N} \\ \text{O} \\ \text{P} \\ \text{Q} \\ \text{R}\end{bmatrix} &
#   \begin{bmatrix}\text{S} \\ \text{T} \\ \text{U} \\ \text{V} \\ \text{W} \\ \text{X}\end{bmatrix}
#   \end{bmatrix}
#
# Batching enables more parallelizable processing. However, batching means that
# the model treats each column independently; for example, the dependence of
# ``G`` and ``F`` can not be learned in the example above.
#

def data_process(raw_text_iter: dataset.IterableDataset, vocab, tokenizer) -> Tensor:
  """Converts raw text into a flat Tensor."""
  data = [torch.tensor(vocab(tokenizer(item)),dtype=torch.long) for item in raw_text_iter]
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data: Tensor, bsz: int) -> Tensor:
  """
  Divides the data into bsz separate sequences, removing extra elements that wouldn't cleanly fit.
  Args:
    data: Tensor, shape [N]
    bsz: int, batch size
  Returns:
    Tensor of shape [N // bsz, bsz]
  """
  seq_len = data.size(0) // bsz
  data = data[:seq_len*bsz]
  data = data.view(bsz,seq_len).t().contiguous()
  return data.to(device)

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def create_wikitext_dataset(params):

  # training dataset
  train_iter = WikiText2(split='train')
  tokenizer = get_tokenizer('basic_english')
  vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
  vocab.set_default_index(vocab['<unk>']) 

  params['ntokens'] = len(vocab)
  
  # train_iter was "consumed" by the process of building the vocab, so we have to create it again
  # I have no idea what that means!!
  train_iter, val_iter, test_iter = WikiText2()
  train_data = data_process(train_iter,vocab,tokenizer)
  val_data = data_process(val_iter,vocab,tokenizer)
  test_data = data_process(test_iter,vocab,tokenizer)

  train_data = batchify(train_data,params['batch_size'])
  val_data = batchify(val_data,params['batch_size'])
  test_data = batchify(test_data,params['batch_size'])

  return train_data,test_data,val_data,vocab,tokenizer

######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# ``get_batch()`` generates a pair of input-target sequences for
# the transformer model. It subdivides the source data into chunks of
# length ``bptt``. For the language modeling task, the model needs the
# following words as ``Target``. For example, with a ``bptt`` value of 2,
# we’d get the following two Variables for ``i`` = 0:
#
# Input:       ->     Target:
# [[A G M S]         [[B H N T]
#  [B H N T]]         [C I O U]]
#
# It should be noted that the chunks are along dimension 0, consistent
# with the ``S`` dimension in the Transformer model. The batch dimension
# ``N`` is along dimension 1.
#

def get_batch(source: Tensor, i: int, bptt) -> Tuple[Tensor,Tensor]:
  """
  Args
    source: Tensor, shape [full_seq_len, batch_size]
    i: int
  Returns:
    tuple (data, target), where data has shape [seq_len,batch_size] and target has shape [seq_len*batch_size]
  """
  seq_len = min(bptt, len(source) - 1 - i)
  data = source[i:i+seq_len] # seq_len x batch_size
  # not sure why we need the reshape here? 
  target = source[i+1:i+seq_len+1].reshape(-1) # seq_len*batch_size
  return data,target

######################################################################
# Train the model
# -------------
#


######################################################################
# We use `CrossEntropyLoss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`__
# with the `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`__
# (stochastic gradient descent) optimizer. The learning rate is initially set to
# 5.0 and follows a `StepLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html>`__
# schedule. During training, we use `nn.utils.clip_grad_norm\_ <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>`__
# to prevent gradients from exploding.
#

import copy
import time

# train for one epoch
def train_epoch(model: nn.Module, criterion, optimizer, train_data, params, epoch, lr) -> None:
  model.train() # turn on train mode
  total_loss = 0.
  log_interval = 200
  start_time = time.time()
  src_mask = generate_square_subsequent_mask(params['bptt']).to(device)

  # is this just one epoch of training?
  num_batches = math.ceil(len(train_data)/params['bptt'])
  batch_train_loss = torch.zeros(num_batches)
  print(f'num_batches = {num_batches}')
  for batch,i in enumerate(range(0,train_data.size(0)-1,params['bptt'])):
    data,targets = get_batch(train_data,i,params['bptt'])
    seq_len = data.size(0)
    if seq_len != params['bptt']: # last batch
      src_mask = src_mask[:seq_len,:seq_len]
    output = model(data,src_mask)
    loss = criterion(output.view(-1,params['ntokens']),targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
    optimizer.step()

    batch_train_loss[batch] = loss.item()
    total_loss += loss.item()
    if batch % log_interval == 0 and batch > 0:
      ms_per_batch = (time.time()-start_time)*1000/log_interval
      cur_loss = total_loss / log_interval
      ppl = math.exp(cur_loss)
      print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
            f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} |'
            f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
      total_loss = 0
      start_time = time.time()

  return batch_train_loss

def evaluate(model: nn.Module, criterion, eval_data: Tensor, params: dict) -> float:
  model.eval() # turn on evaluation mode
  total_loss = 0.
  src_mask = generate_square_subsequent_mask(params['bptt']).to(device)
  with torch.no_grad():
    for i in range(0,eval_data.size(0)-1,params['bptt']):
      data,targets = get_batch(eval_data,i,params['bptt'])
      seq_len = data.size(0)
      if seq_len != params['bptt']: # last batch
        src_mask = src_mask[:seq_len,:seq_len]

      output = model(data, src_mask)
      output_flat = output.view(-1, params['ntokens'])
      # why do we multiply by seq_len here but not in the training criterion? 
      total_loss += seq_len * criterion(output_flat, targets).item()

  # I guess because we divide by total length here
  # slight difference in weight for last batch, but negligible
  return total_loss / (len(eval_data)-1)

######################################################################
# The model hyperparameters are defined below. The vocab size is
# equal to the length of the vocab object.
#

def set_params_attn_is_all_you_need():
  params = set_default_params()
  params['emsize'] = 2048
  params['d_hid'] = 512
  params['nlayers'] = 6
  params['nhead'] = 8
  params['dropout'] = 0.1
  params['epochs'] = 500
  params['batch_size'] = 512
  params['eval_batch_size'] = 256
  params['checkpoint_epochs'] = 5
  return params

def set_default_params(params={}):
  if not 'emsize' in params:
    params['emsize'] = 200 # embedding dimension
  if not 'd_hid' in params:
    params['d_hid'] = 200 #dimension of the feedforward network model in nn.TransformerEncoder
  if not 'nlayers' in params:
    params['nlayers'] = 2 # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
  if not 'nhead' in params:
    params['nhead'] = 2 # number of head in nn.MultiheadAttention
  if not 'dropout' in params:
    params['dropout'] = 0.2 # dropout probability
  if not 'bptt' in params:
    params['bptt'] = 35
  if not 'batch_size' in params:
    params['batch_size'] = 20
  if not 'eval_batch_size' in params:
    params['eval_batch_size'] = 10
  if not 'lr' in params:
    params['lr'] = 5.0 # learning rate
  if not 'epochs' in params:
    params['epochs'] = 3
  if not 'checkpoint_epochs' in params:
    params['checkpoint_epochs'] = 10

  return params

def train(model,criterion,train_data,val_data,params,savedir=None):

  ######################################################################
  # Loop over epochs. Save the model if the validation loss is the best
  # we've seen so far. Adjust the learning rate after each epoch.

  starttime = datetime.datetime.now()
  starttime = starttime.strftime('%Y%m%H%M%S')

  best_val_loss = float('inf')
  best_model = None

  optimizer = torch.optim.SGD(model.parameters(),lr=params['lr'])
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1.0,gamma=.95)
  train_loss = None
    
  val_loss = torch.zeros(params['epochs'])
  val_loss[:] = torch.nan
  for epoch in range(0,params['epochs']):
    epoch_start_time = time.time()
    lr = scheduler.get_last_lr()[0]

    train_loss_curr = train_epoch(model,criterion,optimizer,train_data,params,epoch,lr)
    if train_loss is None:
      train_loss = torch.zeros((len(train_loss_curr),params['epochs']))
      train_loss[:] = torch.nan
    train_loss[:,epoch] = train_loss_curr
    val_loss_curr = evaluate(model,criterion,val_data,params)
    val_loss[epoch] = val_loss_curr
    val_ppl = math.exp(val_loss_curr)
    elapsed = time.time() - epoch_start_time
    print('-'*89)
    print(f'| end of epoch {epoch+1:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss_curr:5.2f} | valid ppl {val_ppl:8.2f}')
    if savedir is not None and ((epoch+1) % params['checkpoint_epochs']) == 0:
      save_model_epoch(savedir,epoch,starttime,net=model,params=params,
        optimizer=optimizer,scheduler=scheduler,
        loss={'best_val': best_val_loss,'train': train_loss,'val': val_loss})

    print('-'*89)
    if val_loss_curr < best_val_loss:
      best_val_loss = val_loss_curr
      best_model = copy.deepcopy(model)
    scheduler.step()

  return best_model,best_val_loss,train_loss,val_loss

def save_model(savefile,net,params,optimizer=None,scheduler=None,loss=None):
  tosave = {'net':net.state_dict(),'params':params}
  if optimizer is not None:
    tosave['optimizer'] = optimizer.state_dict()
  if scheduler is not None:
    tosave['scheduler'] = scheduler.state_dict()
  if loss is not None:
    tosave['loss'] = loss
  torch.save(tosave,savefile)
  return

def save_model_epoch(savedir,epoch,ts,**kwargs):
  savefile = os.path.join(savedir,f"net_epoch{epoch+1:03d}_{ts}.pth")
  print(f'Saving network after training for {epoch} epochs to file {savefile}')
  save_model(savefile,**kwargs)

def load_model(loadfile,net,params,optimizer=None,scheduler=None):
  
  print(f'Loading model from file {loadfile}...')
  state = torch.load(loadfile, map_location=device)
  if net is not None:
    net.load_state_dict(state['net'])
  if params is not None and ('params' in state):
    for k,v in state['params'].items():
      params[k] = v
  if optimizer is not None and ('optimizer' in state):
    optimizer.load_state_dict(state['optimizer'])
  if scheduler is not None and ('scheduler' in state):
    scheduler.load_state_dict(state['scheduler'])
  loss = None
  if 'loss' in state:
    loss = state['loss']

  return loss

def generate_text(prompt_tokens,model,vocab,params,ngenerate=100):
  model.eval() # turn on evaluation mode
  nprompt = len(prompt_tokens)
  maxlen = params['bptt']
  src_mask = generate_square_subsequent_mask(maxlen).to(device)
  with torch.no_grad():
    x = torch.zeros((nprompt+ngenerate,1),dtype=prompt_tokens.dtype).to(device)
    x[:nprompt,0] = prompt_tokens
    for i in range(ngenerate):
      i1 = i+nprompt
      #lencurr = np.minimum(maxlen,nprompt+i)
      #i0 = i1-lencurr
      
      out = model(x[:i1],None)
      w = out[-1].squeeze().exp().cpu()
      w[0] = 0 # do not select unk 
      sampleidx = torch.multinomial(w,1)[0]
      #x[i1] = torch.argmax(out[1:])+1
      x[i1] = sampleidx
    sentence = tokens_to_sentence(x,vocab)
  return sentence,x

def tokens_to_sentence(tokens,vocab):
  if isinstance(tokens,torch.Tensor):
    tokens = tokens.squeeze().cpu().tolist()
  wordlist = vocab.lookup_tokens(tokens)
  sentence = "".join(list(map(lambda x: x+' ',wordlist)))

  return sentence

def sentence_to_tokens(sentence,vocab,tokenizer):
  tokens = torch.tensor(vocab(tokenizer(sentence)),dtype=torch.long)
  return tokens

def main():

  torch.manual_seed(0)
  savedir = os.path.join(os.getcwd(),'wikitext_transformer')
  dosave = True
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  #loadfile = os.path.join(savedir,'net_epoch101_202211153814.pth')
  loadfile = os.path.join(savedir,'net_epoch220_202211003133.pth')
  #loadfile = None

  #params = set_default_params()
  #params['epochs'] = 100
  params = set_params_attn_is_all_you_need()

  train_data,test_data,val_data,vocab,tokenizer = create_wikitext_dataset(params)

  model = TransformerModel(params['ntokens'],params['emsize'],params['nhead'],
                           params['d_hid'],params['nlayers'],params['dropout']).to(device)
  criterion = nn.CrossEntropyLoss()

  if loadfile is None:
    loss = {}
    best_model,loss['best_val'],loss['train'],loss['val'] = \
      train(model,criterion,train_data,val_data,params,savedir=savedir)
    if dosave:
      ts = datetime.datetime.now()
      ts = ts.strftime('%Y%m%H%M%S')
      savefile = os.path.join(savedir,f"net_epoch{params['epochs']+1:03d}_{ts}.pth")
      print(f'Saving network after training to file {savefile}')
      save_model(savefile,best_model,params,loss=loss)

  else:
    loss = load_model(loadfile,model,params)
    best_model = model


  train_iter, val_iter, test_iter = WikiText2()
  test_raw = [item for item in test_iter]
  test_tokens = [sentence_to_tokens(x,vocab,tokenizer) for x in test_raw]


  nprompt = 10
  min_sentence_len = nprompt + 10
  ntest = 10

  sentence_lens = np.array([len(x) for x in test_tokens])
  sentenceidx = np.where(sentence_lens>=min_sentence_len)[0]
  testidx = sentenceidx[np.random.choice(len(sentenceidx),ntest)]
  
  for i in testidx:
    prompt_sentence_full = test_raw[i]
    prompt_tokens_full = test_tokens[i]
    prompt_tokens = prompt_tokens_full[:nprompt]
    prompt_sentence = tokens_to_sentence(prompt_tokens,vocab)
    ngenerate = len(prompt_tokens_full)-len(prompt_tokens)
    gen_sentence,gen_tokens = generate_text(prompt_tokens,model,vocab,params,ngenerate)
    print(f'i = {i}, prompt: {prompt_sentence}')
    print(f'true: {prompt_sentence_full}')
    print(f'generated: {gen_sentence}')
    print('')

  # plot the training and validation loss
  plt.figure()
  plt.plot(loss['train'].cpu().T.flatten(),'.-')
  plt.plot(loss['train'].shape[0]*(1+np.arange(params['epochs'])),loss['val'].cpu(),'.-')
  plt.gca().set_xlabel('Batch')
  plt.gca().set_ylabel('Loss')
  plt.legend(['Train','Val'])
  plt.show()

  ######################################################################
  # Evaluate the best model on the test dataset
  # -------------------------------------------
  #

  test_loss = evaluate(best_model,criterion,test_data,params)
  test_ppl = math.exp(test_loss)
  print('='*89)
  print(f' End of training | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f}')
  print('='*89)


if __name__ == '__main__':
  main()
# %%
