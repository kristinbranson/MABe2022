import os
from datetime import datetime
import numpy as np
from tqdm import tqdm # waitbar
import re
import pickle
import logging

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

def TrainDriver():
  # file containing the data
  xfile = os.path.join(datadir,'Xtesttrain_seq.npy')
  yfile = os.path.join(datadir,'ytesttrain_seq.npy')

  if not os.path.exists(rootsavedir):
    os.mkdir(rootsavedir)
  savedir = os.path.join(rootsavedir,'unet')
  now = datetime.now()
  timestamp = now.strftime('%Y%m%dT%H%M%S')
  if not os.path.exists(savedir):
    os.mkdir(savedir)
  checkpointdir = os.path.join(savedir,'UNet'+timestamp)
  os.mkdir(checkpointdir)
  lossfile = os.path.join(checkpointdir,'alllosses.pkl')
  logfile = os.path.join('train.log')
  logging.basicConfig(filename=logfile,level=logging.DEBUG)
  logging.getLogger().addHandler(logging.StreamHandler())
  
  # hyperparameters
  ntgtsout = 8 # number of flies to keep -- helps us avoid nans
  batch_size = 12
  nepochs = 3   # number of times to cycle through all the data during training
  learning_rate = 0.001 # initial learning rate
  weight_decay = 1e-8 # how learning rate decays over time
  momentum = 0.9 # how much to use previous gradient direction
  nepochs_per_save = 1 # how often to save the network
  niters_per_store = 5 # how often to store losses

  # some day we might want to be able to reload
  loadepoch = 0

  assert(os.path.exists(xfile))
  assert(os.path.exists(yfile))
  
  # load data and initialize Dataset object
  train_dataset = mabe.FlyDataset(xfile,yfile,ntgtsout=ntgtsout)
  train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
  
  # Instantiate the network
  nchannels_inc = int(2**np.round(1.+np.log2(train_dataset.nfeatures*train_dataset.d)))
  net = models.UNet(n_channels=train_dataset.nfeatures*train_dataset.d,n_targets=ntgtsout,nchannels_inc=nchannels_inc,n_categories=train_dataset.ncategories)
  net.to(device=device) # have to be careful about what is done on the CPU vs GPU
  
  # # try the network out before training
  # batch = next(iter(train_dataloader))
  # with torch.no_grad():
  #   pred0 = net.output(batch['x'].to(device=device),verbose=True)
  #   pred0 = pred0[...,0]
  #   type(pred0)
  #   pred0n = pred0.cpu().numpy()
  # plot_annotations(train_dataset,pred0n)
    
  # gradient descent flavor
  optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

  # Following https://github.com/milesial/Pytorch-UNet
  # Use binary cross entropy loss combined with sigmoid output activation function.
  # We combine here for numerical improvements
  #criterion = nn.BCEWithLogitsLoss()
  # inputs: (input, target, weight)
  #criterion = models.BCEWithLogitsWeightedLoss()
  criterion = F.binary_cross_entropy_with_logits
  
  loadfile = None
  loadepoch = 0
  alllosses = None
  #savefile = None
  if loadfile is not None:
    net.load_state_dict(
      torch.load(loadfile,map_location=device)
    )
    m = re.search('[^\d](?P<epoch>\d+)\.pth$',loadfile)
    if m is None:
      logging.warning('Could not parse epoch from file name')
    else:
      loadepoch = int(m['epoch'])
      logging.debug('Parsed epoch from loaded net file name: %d'%loadepoch)
    net.to(device=device)
    if os.path.exists(lossfile):
      with open(lossfile, 'rb') as fid:
        alllosses = pickle.load(fid)
  
  # when we last saved the network
  saveepoch = None

  # how many gradient descent updates we have made
  iters = loadepoch*len(train_dataloader)

  # loop through entire training data set nepochs times
  if alllosses is None:
    alllosses = {'iters': -100*np.ones((nepochs*len(train_dataloader))//niters_per_store), 'epochs': -100*np.ones(nepochs)}
  for epoch in range(loadepoch,nepochs):
    net.train() # put in train mode (affects batchnorm)
    epoch_loss = 0
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
        assert torch.isnan(loss) == False , 'loss is nan, iter %d'%iters
        if iters % niters_per_store == 0:
          alllosses['iters'][iters//niters_per_store] = loss

        epoch_loss += loss.item()
        normloss = loss.item() / torch.count_nonzero(mask)
        pbar.set_postfix(**{'norm loss (batch)': normloss})
        # gradient descent
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()
        iters+=1
  
        pbar.update(x.shape[0])
        
    logging.info('loss (epoch %d) = %f'%(epoch,epoch_loss))
    alllosses['epochs'][epoch] = epoch_loss
    
    # save checkpoint networks every now and then
    if epoch % nepochs_per_save == 0:
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
      torch.save(net.state_dict(),savefile)
      
      with open(lossfile, 'wb') as fid:
        pickle.dump(alllosses, fid, protocol=pickle.HIGHEST_PROTOCOL)
      
  plt.plot(alllosses['iters'])
  plt.show()
  #
  #   # save checkpoint networks every now and then
  #   if epoch % nepochs_per_save == 0:
  #     print('Saving network state at epoch %d'%(epoch+1))
  #     # only keep around the last two epochs for space purposes
  #     if saveepoch is not None:
  #       savefile0 = os.path.join(checkpointdir,f'CP_latest_epoch{saveepoch+1}.pth')
  #       savefile1 = os.path.join(checkpointdir,f'CP_prev_epoch{saveepoch+1}.pth')
  #       if os.path.exists(savefile0):
  #         try:
  #           os.rename(savefile0,savefile1)
  #         except:
  #           print('Failed to rename checkpoint file %s to %s'%(savefile0,savefile1))
  #     saveepoch = epoch
  #     savefile = os.path.join(checkpointdir,f'CP_latest_epoch{saveepoch+1}.pth')
  #     torch.save(net.state_dict(),os.path.join(checkpointdir,f'CP_latest_epoch{epoch + 1}.pth'))
  #
  # torch.save(net.state_dict(),os.path.join(checkpointdir,f'Final_epoch{epoch+1}.pth'))

def PlotTrainLoss(lossfile=None):
  if lossfile is None:
    savedir = os.path.join(rootsavedir,'unet')
    checkpointdir = os.path.join(savedir,'UNet20220124T120509')
    lossfile = os.path.join(checkpointdir,'alllosses.pkl')
  with open(lossfile, 'rb') as fid:
    alllosses = pickle.load(fid)
  plt.plot(alllosses['iters'])
  plt.show()
  
if __name__ == "__main__":
  #PlotTrainLoss()
  TrainDriver()