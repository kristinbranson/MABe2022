import os
from datetime import datetime
import numpy as np
from tqdm import tqdm # waitbar
import re
import pickle
import logging
import glob

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm,animation,colors

import MABeFlyUtils as mabe
import models

print('CUDA available: %d'%torch.cuda.is_available())
#device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# local path to data -- modify this!
datadir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/seqdata20220118'
rootsavedir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/savednets'

def plot_annotations(train_dataset,pred0n):
  # plot
  yvalues = [0,1] # unique y-values
  ycolors = cm.get_cmap('tab20')
  ycolors = ycolors(np.linspace(0.,1.,20))
  ycolors = np.tile(ycolors,(int(np.ceil(train_dataset.ncategories/20)),1))
  ycolors = ycolors[:train_dataset.ncategories,:]
  pred0nb = pred0n > .5
  mabe.plot_annotations(pred0nb[0,:,:],values=yvalues,patchcolors=ycolors,binarycolors=True,names=train_dataset.categorynames)
  plt.show()

def SetLoggingState(logfile):
  logging.basicConfig(filename=logfile,level=logging.INFO)
  logging.getLogger().addHandler(logging.StreamHandler())
  logging.getLogger('matplotlib.font_manager').disabled = True
  return()

def LoadState(loadfile,net,optimizer,scheduler):
  
  loadepoch = 0
  alllosses = {'iters': {}, 'epochs': {}}
  if loadfile is None:
    return loadepoch,alllosses
    
  logging.info('Loading checkpoint...')
  state = torch.load(loadfile, map_location=device)
  net.load_state_dict(state['net'])
  optimizer.load_state_dict(state['optimizer'])
  scheduler.load_state_dict(state['scheduler'])
  alllosses = state['alllosses']
  
  m = re.search('[^\d](?P<epoch>\d+)\.pth$', loadfile)
  if m is None:
    logging.warning('Could not parse epoch from file name')
  else:
    loadepoch = int(m['epoch'])
    logging.debug('Parsed epoch from loaded net file name: %d'%loadepoch)
  net.to(device=device)
  logging.info('Done')
  
  return loadepoch,alllosses

def SaveCheckpoint(epoch,net,optimizer,scheduler,alllosses,saveepoch,checkpointdir):
  logging.info('Saving network state at epoch %d'%(epoch+1))
  # only keep around the last two epochs for space purposes
  if saveepoch is not None:
    savefile0 = os.path.join(checkpointdir,f'CP_latest_epoch{saveepoch+1}.pth')
    savefile1 = os.path.join(checkpointdir,f'CP_prev_epoch{saveepoch+1}.pth')
    if os.path.exists(savefile0):
      try:
        os.rename(savefile0,savefile1)
      except:
        logging.warning('Failed to rename checkpoint file %s to %s'%(savefile0,savefile1))
  saveepoch = epoch
  savefile = os.path.join(checkpointdir,f'CP_latest_epoch{saveepoch+1}.pth')
  torch.save({'net':net.state_dict(),'optimizer':optimizer.state_dict(),
              'scheduler':scheduler.state_dict(),'alllosses':alllosses},savefile)
  return saveepoch,savefile

  
def TrainDriver(loadfile=None):
  # file containing the data
  xfile = os.path.join(datadir,'Xtesttrain_seq.npy')
  yfile = os.path.join(datadir,'ytesttrain_seq.npy')
  assert(os.path.exists(xfile))
  assert(os.path.exists(yfile))
  
  checkpointdir = SetCheckpointDir()
  
  logfile = os.path.join(checkpointdir,'train.log')
  SetLoggingState(logfile)
  
  # hyperparameters
  ntgtsout = 8 # number of flies to keep -- helps us avoid nans
  batch_size = 12
  nepochs = 1000   # number of times to cycle through all the data during training
  learning_rate = 0.001 # initial learning rate
  weight_decay = 1e-8 # how learning rate decays over time
  momentum = 0.9 # how much to use previous gradient direction
  nepochs_per_save = 1 # how often to save the network
  niters_per_store = 5 # how often to store losses

  # load data and initialize Dataset object
  logging.info('Loading dataset...')
  train_dataset = mabe.FlyDataset(xfile,yfile,ntgtsout=ntgtsout)
  train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
  logging.info('Done.')
  
  # Instantiate the network
  logging.info('Creating network...')
  nchannels_inc = int(2**np.round(1.+np.log2(train_dataset.nfeatures*train_dataset.d)))
  net = models.UNet(n_channels=train_dataset.nfeatures*train_dataset.d,n_targets=ntgtsout,nchannels_inc=nchannels_inc,n_categories=train_dataset.ncategories)
  net.to(device=device) # have to be careful about what is done on the CPU vs GPU
  logging.info('Done.')
  
  # gradient descent flavor
  optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
  
  # # try the network out before training
  # batch = next(iter(train_dataloader))
  # with torch.no_grad():
  #   pred0 = net.output(batch['x'].to(device=device),verbose=True)
  #   pred0 = pred0[...,0]
  #   type(pred0)
  #   pred0n = pred0.cpu().numpy()
  # plot_annotations(train_dataset,pred0n)
  
  # Following https://github.com/milesial/Pytorch-UNet
  # Use binary cross entropy loss combined with sigmoid output activation function.
  # We combine here for numerical improvements
  #criterion = nn.BCEWithLogitsLoss()
  criterion = F.binary_cross_entropy_with_logits
  
  loadepoch,alllosses = LoadState(loadfile,net,optimizer,scheduler)

  # how many gradient descent updates we have made
  iters = loadepoch*len(train_dataloader)
  nbatches_per_epoch = len(train_dataloader)

  # when we last saved the network
  saveepoch = None

  # prepare for plotting loss
  hloss = None
  axloss = None
  figloss = None
  plt.ion()

  logging.info('Training...')

  # loop through entire training data set nepochs times
  for epoch in range(loadepoch,nepochs):
    net.train() # put in train mode (affects batchnorm)
    epoch_loss = 0
    iter_loss = 0.
    with tqdm(total=len(train_dataset),desc=f'Epoch {epoch + 1}/{nepochs}',unit='img') as pbar:
      # loop through each batch in the training data
      for batch in train_dataloader:
        # compute the loss
        x = batch['x']
        assert torch.any(torch.isnan(x)) == False ,'x contains nans, iter %d'%iters
        x = x.to(device=device, dtype=torch.float32) # transfer to GPU
        y = batch['y']
        y = y.to(device=device, dtype=torch.float32) # transfer to GPU
        w = batch['weights']
        w = w.to(device=device, dtype=torch.float32) # transfer to GPU
        ypred = net(x) # evaluate network on batch
        assert torch.any(torch.isnan(ypred)) == False , 'ypred contains nans, iter %d'%iters
        mask = torch.isnan(y)==False
        loss = criterion(ypred[mask][...,0],y[mask],w[mask]) # compute loss
        loss_scalar = loss.detach().item()
        assert torch.isnan(loss) == False , 'loss is nan, iter %d'%iters
        
        # gradient descent
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()
        
        # store loss for record
        pbar.set_postfix(**{'loss (batch)': loss_scalar})
        iter_loss += loss_scalar
        epoch_loss += loss_scalar
        if iters % niters_per_store == 0:
          alllosses['iters'][iters] = loss_scalar
          iter_loss = 0.
        
        iters+=1
        pbar.update(x.shape[0])
        
    logging.info('loss (epoch %d) = %f'%(epoch,epoch_loss/nbatches_per_epoch))
    alllosses['epochs'][epoch] = epoch_loss/nbatches_per_epoch
    alllosses['nbatches_per_epoch'] = nbatches_per_epoch
    
    # save checkpoint networks every now and then
    if epoch % nepochs_per_save == 0:
      saveepoch,savefile = SaveCheckpoint(epoch,net,optimizer,scheduler,alllosses,saveepoch,checkpointdir)
      
      # update plots
      if figloss is not None and not plt.fignum_exists(figloss.number):
        hloss = None
        axloss = None
        figloss = None
      hloss,axloss,figloss = PlotAllLosses(alllosses,h=hloss,ax=axloss,fig=figloss)
      plt.draw()
      plt.pause(.001)
      
  torch.save({'net':net.state_dict(),'optimizer':optimizer.state_dict(),
              'scheduler':scheduler.state_dict(),'alllosses':alllosses},savefile)
  PlotAllLosses(alllosses,h=hloss,ax=axloss)
  plt.show()

def PlotAllLosses(alllosses,h=None,ax=None,fig=None):
  isnewaxis = False
  if h is None:
    h = {}
  if ax is None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    isnewaxis = True
  if 'iters' in h:
    hcurr = h['iters']
  else:
    hcurr = None
  h['iters'],ax = PlotLoss(alllosses['iters'],ax=ax,label='Batch loss',h=hcurr)
  ax.set_xlabel('Iterations')
  ax.set_ylabel('Loss')
  if 'epochs' in h:
    hcurr = h['epochs']
  else:
    hcurr = None
  h['epochs'],ax = PlotLoss(alllosses['epochs'],ax=ax,multiplier=alllosses['nbatches_per_epoch'],label='Epoch loss',h=hcurr)
  ax.set_yscale('log')
  _ = ax.axis('tight')
  if isnewaxis:
    ax.legend()
  return h,ax,fig
  
def PlotLoss(lossdict,ax=None,multiplier=1.,label=None,h=None):
  if ax is None and h is None:
    ax = plt.subplot(111)
  iters0 = np.array(list(lossdict.keys()))
  iters1 = iters0[1:]
  iters1 = np.append(iters1,iters0[-1]+1.)
  x = list(lossdict.values())[0]
  if type(x) == float:
    losses = np.array(list(lossdict.values()))
  else:
    losses = np.array(list(map( lambda x: x.item(), lossdict.values())))
  iters = np.vstack((iters0,iters1)).T.flatten()*multiplier
  losses = np.vstack((losses,losses)).T.flatten()
  if h is None:
    h = ax.plot(iters,losses,label=label)[0]
  else:
    h.set_data(iters,losses)
  return h,ax

def GetLatestCheckpoint():
  savedir = os.path.join(rootsavedir,'unet')
  checkpointdir = None
  checkpointdate = ''
  netfile = None
  for cpd in os.scandir(savedir):
    if cpd.is_dir():
      m = re.search('UNet(\d{8}T\d{6})',cpd.path)
      if m is None:
        continue
      nf = glob.glob(os.path.join(cpd,'CP_latest_epoch*.pth'))
      if len(nf) == 0:
        continue
      if m.group(1) > checkpointdate:
        checkpointdate = m.group(1)
        checkpointdir = cpd.path
        netfile = nf[0]
  return checkpointdir,netfile
  
def SetCheckpointDir():
  if not os.path.exists(rootsavedir):
    os.mkdir(rootsavedir)
  savedir = os.path.join(rootsavedir,'unet')
  now = datetime.now()
  timestamp = now.strftime('%Y%m%dT%H%M%S')
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  checkpointdir = os.path.join(savedir,'UNet'+timestamp)
  os.mkdir(checkpointdir)
  return checkpointdir
  
def PlotTrainLoss(loadfile=None):
  if loadfile is None:
    _, loadfile = GetLatestCheckpoint()
  state = torch.load(loadfile, map_location=device)
  alllosses = state['alllosses']

  PlotAllLosses(alllosses)
  plt.show()
  
if __name__ == "__main__":
  #PlotTrainLoss()
  #loadfile = None
  checkpointdir,loadfile = GetLatestCheckpoint()
  TrainDriver(loadfile=loadfile)