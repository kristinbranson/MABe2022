import os
from datetime import datetime
import numpy as np
from tqdm import tqdm # waitbar
import re
import pickle
import logging
import glob
import shutil
import scipy

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

#print('CUDA available: %d'%torch.cuda.is_available())
#device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# local path to data -- modify this!
datadir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/seqdata20220307'
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

def LoadState(loadfile, net=None, optimizer=None, scheduler=None, config=None):
  
  loadepoch = 0
  alllosses = {'iters': {}, 'epochs': {}, 'holdout': {}, 'train': {}}
  if loadfile is None:
    return loadepoch,alllosses
    
  logging.info('Loading checkpoint...')
  state = torch.load(loadfile, map_location=device)
  if net is not None:
    net.load_state_dict(state['net'])
  if optimizer is not None:
    optimizer.load_state_dict(state['optimizer'])
  if scheduler is not None:
    scheduler.load_state_dict(state['scheduler'])
  if config is not None and ('config' in state):
    for k,v in state['config'].items():
      config[k] = v
    
  alllosses = state['alllosses']
  
  m = re.search('[^\d](?P<epoch>\d+)\.pth$', loadfile)
  if m is None:
    logging.warning('Could not parse epoch from file name')
  else:
    loadepoch = int(m['epoch'])
    logging.debug('Parsed epoch from loaded net file name: %d'%loadepoch)
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

def CleanOldNets(checkpointdir=None,deleteolddirs=False,deleteoldfiles=False):
  deleted = []
  savedir = os.path.join(rootsavedir,'unet')
  lastcheckpointdir,_ = GetLatestCheckpoint()
  
  if deleteolddirs:
    for cpd in os.scandir(savedir):
      if not cpd.is_dir():
        continue
      if cpd.path == lastcheckpointdir:
        continue
      else:
        logging.info('Deleting old checkpoint directory %s'%cpd.path)
        shutil.rmtree(cpd.path)
        deleted.append(cpd.path)
        
  if checkpointdir is None:
    checkpointdir = lastcheckpointdir
  else:
    if checkpointdir in deleted:
      return deleted
      
  if deleteoldfiles:
    oldfiles = glob.glob(os.path.join(lastcheckpointdir,'CP_prev_epoch*.pth'))
    for oldfile in oldfiles:
      logging.info('Deleting checkpoint network %s'%oldfile)
      os.remove(oldfile)
      deleted.append(oldfile)

  return deleted

def CombineLims(xlims,xlims1):
  xlims = (np.minimum(xlims[0],xlims1[0]),np.maximum(xlims[1],xlims1[1]))
  return xlims

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
  h['iters'],ax,xlims,ylims = PlotLoss(alllosses['iters'],ax=ax,multiplier=1.,label='Batch training loss',h=hcurr,setlims=True)
  ax.set_xlabel('Iterations')
  ax.set_ylabel('Loss')
  if 'epochs' in h:
    hcurr = h['epochs']
  else:
    hcurr = None
  h['epochs'],ax,xlims1,ylims1 = PlotLoss(alllosses['epochs'],ax=ax,multiplier=alllosses['nbatches_per_epoch'],label='Epoch training loss',h=hcurr)
  xlims = CombineLims(xlims,xlims1)
  ylims = CombineLims(ylims,ylims1)

  if 'holdout' in alllosses and alllosses['holdout'] is not None and len(alllosses['holdout']) > 0:
    if 'holdout' in h:
      hcurr = h['holdout']
    else:
      hcurr = None
    h['holdout'],ax,xlims1,ylims1 = PlotLoss(alllosses['holdout'],ax=ax,multiplier=alllosses['nbatches_per_epoch'],label='Epoch holdout loss',h=hcurr)
    xlims = CombineLims(xlims,xlims1)
    ylims = CombineLims(ylims,ylims1)

  _ = ax.set_xlim(xlims)
  _ = ax.set_ylim(ylims)

  # if 'train' in alllosses and alllosses['train'] is not None and len(alllosses['train']) > 0:
  #   if 'train' in h:
  #     hcurr = h['train']
  #   else:
  #     hcurr = None
  #   h['train'],ax = PlotLoss(alllosses['train'],ax=ax,multiplier=alllosses['nbatches_per_epoch'],label='Epoch holdout loss',h=hcurr)

  if isnewaxis:
    ax.legend()
  return h,ax,fig

def Evaluate(net,dataloader,tonumpy=False):

  net.eval()
  with torch.no_grad():
    with tqdm(total=len(dataloader)*dataloader.batch_size,unit='seq') as pbar:
      for batch in dataloader:
        x = batch['x']
        x = x.to(device=device, dtype=torch.float32) # transfer to GPU
        y = batch['y']
        y = y.to(device=device, dtype=torch.float32) # transfer to GPU
        pred = net.output(x)
        if tonumpy:
          pred = pred.cpu().numpy()
        seqnum = y['seqnum']
        ypred = {}
        for i in range(dataloader.batch_size):
          seqid = dataloader.dataset.seqids[seqnum[i]]
          ypred[seqid] = pred[i,...]
        pbar.update(x.shape[0])
  return ypred

def ComputeErrorMetrics(net,dataset,dataloader,ypred=None):
  net.eval()
  
  confusionmatrix = []
  for c in range(dataset.ncategories):
    confusionmatrix.append(np.zeros((len(dataset.yvalues[c]),len(dataset.yvalues[c]))))
    
  if ypred is None:
    
    with torch.no_grad():
      with tqdm(total=len(dataset),unit='seq') as pbar:
        for batch in dataloader:
          x = batch['x']
          x = x.to(device=device, dtype=torch.float32) # transfer to GPU
          y = batch['y']
          y = y.to(device=device, dtype=torch.float32) # transfer to GPU
          pred = net.output(x)
          predbin = pred[...,0] > .5
          for c in range(dataset.ncategories):
            for ipred in range(len(dataset.yvalues[c])):
              vpred = dataset.yvalues[c][ipred]
              for itrue in range(len(dataset.yvalues[c])):
                vtrue = dataset.yvalues[c][itrue]
                confusionmatrix[c][itrue,ipred] += torch.count_nonzero(torch.logical_and(predbin[:,c,:] == vpred,y[:,c,:] == vtrue))
          pbar.update(x.shape[0])
          
  else:
    for seqid,pred in ypred.items():
      y = dataset.y[seqid]
      predbin = pred[...,0] > .5
      for c in range(dataset.ncategories):
        for ipred in range(len(dataset.yvalues[c])):
          vpred = dataset.yvalues[c][ipred]
          for itrue in range(len(dataset.yvalues[c])):
            vtrue = dataset.yvalues[c][itrue]
            confusionmatrix[c][itrue,ipred] += torch.count_nonzero(torch.logical_and(predbin[:,c,:] == vpred,y[:,c,:] == vtrue))

  return confusionmatrix
  
def PlotLoss(lossdict,ax=None,multiplier=1.,label=None,h=None,setlims=False):
  # multiplier: number of iterations of training this corresponds to (x-axis)
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
  xlims = (np.min(iters)-5,np.max(iters)+5)
  ylims = (np.maximum(1e-10,np.min(losses)/1.01),1.01*np.max(losses))
  if setlims:
    ax.set_yscale('log')
    _ = ax.set_xlim(xlims)
    _ = ax.set_ylim(ylims)
  return h,ax,xlims,ylims

def SetConfig(loadfile=None):
  config = {}
  config['ntgtsout'] = 8 # number of flies to keep -- helps us avoid nans
  config['batch_size'] = 12
  config['nepochs'] = 1000   # number of times to cycle through all the data during training
  config['learning_rate'] = 0.001 # initial learning rate
  config['weight_decay'] = 1e-8 # how learning rate decays over time
  config['momentum'] = 0.9 # how much to use previous gradient direction
  config['nepochs_per_save'] = 5 # how often to save the network
  config['niters_per_store'] = 5 # how often to store losses
  config['frac_holdout'] = 0.5 # fraction of sequences to use as a holdout set
  config['holdout_sample'] = 'uniform' # whether to sample uniformly at random, or weight based on rarity of labels
  if loadfile is not None:
    _,_ = LoadState(loadfile, config=config)
  return config

def SetConfigDataset(config, dataset):
  if 'nchannels_inc' not in config:
    config['nchannels_inc'] = int(2**np.round(1.+np.log2(dataset.nfeatures * dataset.d)))
  if 'n_channels' not in config:
    config['n_channels'] = dataset.nfeatures * dataset.d
  if 'n_categories' not in config:
    config['n_categories'] = dataset.ncategories
  return

def InitNet(config):
  net = models.UNet(n_channels=config['n_channels'],n_targets=config['ntgtsout'],
                    nchannels_inc=config['nchannels_inc'],n_categories=config['n_categories'])
  net.to(device=device) # have to be careful about what is done on the CPU vs GPU
  return net

def ComputeLoss(dataloader,net,criterion,desc=None):
  totalloss = 0.
  net.eval()
  
  with tqdm(total=len(dataloader),desc=desc,unit='batch') as pbar:
    # loop through each batch in the training data
    batchi = 0
    for batch in dataloader:
      # compute the loss
      x = batch['x']
      assert torch.any(torch.isnan(x)) == False ,'x contains nans, batch %d'%batchi
      x = x.to(device=device, dtype=torch.float32) # transfer to GPU
      y = batch['y']
      y = y.to(device=device, dtype=torch.float32) # transfer to GPU
      w = batch['weights']
      w = w.to(device=device, dtype=torch.float32) # transfer to GPU
      ypred = net(x) # evaluate network on batch
      assert torch.any(torch.isnan(ypred)) == False , 'ypred contains nans, batch %d'%batchi
      mask = torch.isnan(y)==False
      loss = criterion(ypred[mask][...,0],y[mask],w[mask]) # compute loss
      loss_scalar = loss.detach().item()
      assert torch.isnan(loss) == False , 'loss is nan, batch %d'%batchi
      
      # store loss for record
      pbar.set_postfix(**{'holdout loss (batch)': loss_scalar})
      totalloss += loss_scalar
      pbar.update(1)
      batchi+=1
  return totalloss

def ClassCountDriver():
  config = SetConfig()

  xfilestrs = ['Xusertrain_seq.npy','Xtesttrain_seq.npy','Xtest1_seq.npy','Xtest2_seq.npy']
  yfilestrs = ['yusertrain_seq.npy','ytesttrain_seq.npy','ytest1_seq.npy','ytest2_seq.npy']
  #xfilestrs = ['Xtest2_seq.npy']
  #yfilestrs = ['ytest2_seq.npy']

  newnames = [
    'Control 1',
    'pC1 on vs off',
    'Any aggression',
    'Female vs male',
    'Control 1 sex separated',
    'Control 2',
    '71G01',
    'Male R71G01 female control',
    'R65F12',
    'R91B01',
    'Blind control',
    'aIPg',
    'pC1d',
    'Blind aIPg',
    'Blind control on vs off',
    'Blind control strong vs off',
    'Blind control weak vs off',
    'Blind control strong vs weak',
    'Blind control last vs first',
    'Control 2 on vs off',
    'Control 2 strong vs off',
    'Control 2 weak vs off',
    'Control 2 strong vs weak',
    'Control 2 last vs first',
    'Blind aIPg on vs off',
    'Blind aIPg strong vs off',
    'Blind aIPg weak vs off',
    'Blind aIPg strong vs weak',
    'Blind aIPg last vs first',
    'aIPg on vs off',
    'aIPg strong vs off',
    'aIPg weak vs off',
    'aIPg strong vs weak',
    'aIPg last vs first',
    'pC1d strong vs off',
    'pC1d weak vs off',
    'pC1d strong vs weak',
    'pC1d last vs first',
    'Any courtship',
    'Any control',
    'Any blind',
    'Any aIPg',
    'Any R71G01',
    'Any sex-separated',
    'Aggression manual annotation',
    'Chase manual annotation',
    'Courtship manual annotation',
    'High fence manual annotation',
    'Wing ext.~manual annotation',
    'Wing flick manual annotation',
  ]

  allycounts = []
  allyseqcounts = []
  for filei in range(len(xfilestrs)):
  
    xfile = os.path.join(datadir,xfilestrs[filei])
    yfile = os.path.join(datadir,yfilestrs[filei])
    dataset = mabe.FlyDataset(xfile,yfile,ntgtsout=config['ntgtsout'],arena_radius=mabe.ARENA_RADIUS_MM)
    
    for c in range(dataset.ncategories):
      print(f'{c}: {newnames[c]}, {dataset.categorynames[c]}')
    
    yvalues = dataset.compute_y_values(binary=True)
    ycounts,n = dataset.compute_y_counts(yvalues=yvalues)
    allycounts.append(ycounts)
    yseqcounts,n = dataset.compute_y_seq_counts(yvalues=yvalues)
    allyseqcounts.append(yseqcounts)
    for c in range(dataset.ncategories):
      for v in yvalues[c]:
        print(f'Task {c}: val {v}: {ycounts[c][v]} fr, {yseqcounts[c][v]} seq')
  print('For frame count table')
  for c in range(dataset.ncategories):
    s = f'{newnames[c]}'
    for filei in range(len(xfilestrs)):
      s += f' & {allycounts[filei][c][1]:,}'
    for filei in range(len(xfilestrs)):
      s += f' & {allycounts[filei][c][0]:,}'
    print(s+'\\\\\\hline')
    
  print('For seq count table')
    
  for c in range(dataset.ncategories):
    s = f'{newnames[c]}'
    for filei in range(len(xfilestrs)):
      s += f' & {allyseqcounts[filei][c][1]:,}'
    for filei in range(len(xfilestrs)):
      s += f' & {allyseqcounts[filei][c][0]:,}'
    print(s+'\\\\\\hline')


def TrainDriver(loadfile=None):
  # file containing the data
  xfile = os.path.join(datadir,'Xusertrain_seq.npy')
  yfile = os.path.join(datadir,'yusertrain_seq.npy')
  assert(os.path.exists(xfile))
  assert(os.path.exists(yfile))
  
  checkpointdir = SetCheckpointDir()
  
  logfile = os.path.join(checkpointdir,'train.log')
  SetLoggingState(logfile)
  
  # config
  config = SetConfig(loadfile)

  # load data and initialize Dataset object
  logging.info('Loading dataset...')
  all_dataset = mabe.FlyDataset(xfile,yfile,ntgtsout=config['ntgtsout'],arena_radius=mabe.ARENA_RADIUS_MM)
  
  selectcategories = ['perframe_chase']
  if selectcategories is not None:
    all_dataset.select_categories(selectcategories)
  all_dataset.balance_reweight(binary=True)
  
  isholdout = config['frac_holdout'] > 0.
  if isholdout:
    train_dataset,holdout_dataset,pholdout,idxholdout = mabe.split_train_holdout(all_dataset,config['frac_holdout'],config['holdout_sample'])
  else:
    train_dataset = all_dataset
    
  # preprocess both data sets
  train_dataset.preprocess_all()
  holdout_dataset.preprocess_all()
  
  fig,ax = plt.subplots(3,1,sharex=True)
  maxminnlabels = 0
  for seqnum in range(train_dataset.nseqs):
    minnlabels = np.minimum(np.min(np.sum(train_dataset.y[train_dataset.seqids[seqnum]]==0,axis=1),axis=0),
                            np.min(np.sum(train_dataset.y[train_dataset.seqids[seqnum]]==1,axis=1),axis=0))
    tgt = np.argmax(minnlabels)
    if minnlabels[tgt] > maxminnlabels:
      maxminnlabels = minnlabels[tgt]
      besttgt = tgt
      bestseq = seqnum
      
  mabe.plot_annotations(y=train_dataset.y[train_dataset.seqids[seqnum]][...,tgt],yvalues=train_dataset.yvalues,names=train_dataset.categorynames,binarycolors=True,ax=ax[0])
  idx = train_dataset.seqtgt2idx(seqnum=0,tgt=tgt)
  x = train_dataset.getx(idx)
  for i in range(1,x.shape[3]):
    ax[1].plot(x[mabe.headidx,0,:,i])
  ax[1].plot(x[mabe.headidx,0,:,0],'k-',linewidth=2)
  for i in range(1,x.shape[3]):
    ax[2].plot(x[mabe.headidx,1,:,i])
  ax[2].plot(x[mabe.headidx,1,:,0],'k-',linewidth=2)
  
  # zscore using just training data
  train_dataset.zscore()
  if isholdout:
    zscore_stats = train_dataset.get_zscore_stats()
    holdout_dataset.zscore(set=zscore_stats)
  
  train_dataloader = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True)
  if isholdout:
    holdout_dataloader = DataLoader(holdout_dataset,batch_size=config['batch_size'],shuffle=True)
  else:
    print(f'isholdout = {isholdout}')

  logging.info('Done.')
  
  # Instantiate the network
  logging.info('Creating network...')
  SetConfigDataset(config,train_dataset)
  net = InitNet(config)
  net.to(device=device) # have to be careful about what is done on the CPU vs GPU
  logging.info('Done.')
  
  # gradient descent flavor
  optimizer = optim.RMSprop(net.parameters(), lr=config['learning_rate'],
                            weight_decay=config['weight_decay'],
                            momentum=config['momentum'])
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
  
  # # try the network out before training
  # batch = next(iter(train_dataloader))
  # with torch.no_grad():
  #   pred0 = net.output(batch['x'].to(device=device),verbose=True)
  #   pred0 = pred0[...,0]
  #   type(pred0)
  #   pred0n = pred0.cpu().numpy()
  # plot_annotations(dataset,pred0n)
  
  # Following https://github.com/milesial/Pytorch-UNet
  # Use binary cross entropy loss combined with sigmoid output activation function.
  # We combine here for numerical improvements
  #criterion = nn.BCEWithLogitsLoss()
  criterion = F.binary_cross_entropy_with_logits
  
  loadepoch,alllosses = LoadState(loadfile,net,optimizer,scheduler)

  # how many gradient descent updates we have made
  iters = loadepoch*len(train_dataloader)
  nbatches_per_epoch = len(train_dataloader)
  if isholdout:
    nbatches_holdout = len(holdout_dataloader)

  # when we last saved the network
  saveepoch = None

  # prepare for plotting loss
  hloss = None
  axloss = None
  figloss = None
  plt.ion()

  logging.info('Training...')

  # loop through entire training data set nepochs times
  for epoch in range(loadepoch,config['nepochs']):
    net.train() # put in train mode (affects batchnorm)
    epoch_loss = 0
    iter_loss = 0.
    with tqdm(total=len(train_dataset),desc=f"Epoch {epoch + 1}/{config['nepochs']}",unit='img') as pbar:
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
        if iters % config['niters_per_store'] == 0:
          alllosses['iters'][iters] = iter_loss / config['niters_per_store'] / config['batch_size']
          iter_loss = 0.
        
        iters+=1
        pbar.update(x.shape[0])
    logging.info('train loss (epoch %d) = %f'%(epoch,epoch_loss/nbatches_per_epoch/config['batch_size']))
    alllosses['epochs'][epoch] = epoch_loss/nbatches_per_epoch/config['batch_size']
    alllosses['nbatches_per_epoch'] = nbatches_per_epoch

    if isholdout:
      holdout_loss = ComputeLoss(holdout_dataloader,net,criterion,desc=f"Holdout eval, epoch {epoch + 1}/{config['nepochs']}")
      logging.info('holdout loss (epoch %d) = %f'%(epoch,holdout_loss/nbatches_holdout/config['batch_size']))
      alllosses['holdout'][epoch] = holdout_loss/nbatches_holdout/config['batch_size']
    
    # train_loss = ComputeLoss(train_dataloader,net,criterion,desc=f"Train eval, epoch {epoch + 1}/{config['nepochs']}")
    # logging.info('train loss (epoch %d) = %f'%(epoch,train_loss/nbatches_per_epoch))
    # alllosses['train'][epoch] = train_loss/nbatches_per_epoch/config['batch_size']
    
    # save checkpoint networks every now and then
    if epoch % config['nepochs_per_save'] == 0:
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
              'scheduler':scheduler.state_dict(),'alllosses':alllosses,
              'config':config},savefile)
  PlotAllLosses(alllosses,h=hloss,ax=axloss)
  plt.show()
  
def PlotTrainLossDriver(loadfile=None):
  if loadfile is None:
    _, loadfile = GetLatestCheckpoint()
  state = torch.load(loadfile, map_location=device)
  alllosses = state['alllosses']

  PlotAllLosses(alllosses)
  plt.show()
  
def PrintConfusionMatrix(dataset,confusionmatrix):
  for c in range(dataset.ncategories):
    if len(dataset.yvalues[c]) == 2:
      print('{catname}: FPR = {fpr}, FNR = {fnr}'.format(catname=dataset.categorynames[c],
                                                         fpr=confusionmatrix[c][0,1]/np.sum(confusionmatrix[c][0,:]),
                                                         fnr=confusionmatrix[c][1,0]/np.sum(confusionmatrix[c][1,:])))
    else:
      print('%s:'%dataset.categorynames[c])
      con = confusionmatrix[c] / np.sum(confusionmatrix[c],axis=1)
      print('%s'%str(con))
  return
  
def EvalDriver(loadfile=None):
  
  xtrainfile = os.path.join(datadir, 'Xusertrain_seq.npy')
  ytrainfile = os.path.join(datadir, 'yusertrain_seq.npy')
  xtestafile = os.path.join(datadir, 'Xtesttrain_seq.npy')
  ytestafile = os.path.join(datadir, 'ytesttrain_seq.npy')
  xtestbfile = os.path.join(datadir, 'Xtest1_seq.npy')
  ytestbfile = os.path.join(datadir, 'ytest1_seq.npy')

  assert(os.path.exists(xtrainfile))
  assert(os.path.exists(ytrainfile))
  if loadfile is None:
    _, loadfile = GetLatestCheckpoint()
  config = SetConfig(loadfile)
  dataset = mabe.FlyDataset(xtrainfile, ytrainfile, ntgtsout=config['ntgtsout'], frac_holdout=config['frac_holdout'])
  dataloader = DataLoader(dataset,batch_size=config['batch_size'],shuffle=False)
  logging.info('Creating network...')
  SetConfigDataset(config,dataset)
  net = InitNet(config)
  loadepoch,alllosses = LoadState(loadfile,net=net)
  logging.info('Done.')
  ytypes = mabe.get_ytypes(dataset)
  # trainconfusionmatrix = ComputeErrorMetrics(net,dataset,dataloader,ytypes)
  # print('Train error rates:')
  # PrintConfusionMatrix(dataset,trainconfusionmatrix)

  testdataset = mabe.FlyDataset(xtestafile, ytestafile, ntgtsout=config['ntgtsout'],
                                zscore_set=dataset.get_zscore_stats(),
                                weight_set=dataset.get_weight_stats())
  testdataloader = DataLoader(testdataset,batch_size=config['batch_size'],shuffle=False)
  testaconfusionmatrix = ComputeErrorMetrics(net,testdataset,testdataloader,ytypes)
  print('Test error rates:')
  PrintConfusionMatrix(testdataset,testaconfusionmatrix)

def PoseRepDriver():
  # file containing the data
  xfile = os.path.join(datadir, 'Xusertrain_seq.npy')
  yfile = os.path.join(datadir, 'yusertrain_seq.npy')
  nsample = 500
  assert (os.path.exists(xfile))
  assert (os.path.exists(yfile))

  # load data and initialize Dataset object
  logging.info('Loading dataset...')
  all_dataset = mabe.FlyDataset(xfile, yfile, arena_radius=mabe.ARENA_RADIUS_MM)

  X = next(iter(all_dataset.X.values()))
  sz = X.shape[:2]
  y = next(iter(all_dataset.y.values()))
  maxnflies = X.shape[3]
  T = X.shape[2]
  X = np.zeros((X.shape[0],X.shape[1],nsample*len(all_dataset.X)))
  y = np.zeros((y.shape[0],nsample*len(all_dataset.X)))
  flyid = np.zeros(nsample*len(all_dataset.X),dtype=int)
  flyid0=np.tile(np.arange(maxnflies,dtype=int).reshape(1,maxnflies),(T,1)).flatten()
  count = 0
  flycount = 0
  scale_perfly = np.zeros((len(mabe.scalenames),0))

  for id in all_dataset.seqids:

    Xcurr = all_dataset.X[id]
    ycurr = all_dataset.y[id]

    isreal = mabe.get_real_flies(Xcurr.reshape((Xcurr.shape[0]*Xcurr.shape[1],)+Xcurr.shape[2:]))
    scale_perfly_curr = np.zeros((len(mabe.scalenames),maxnflies))
    scale_perfly_curr[:] = np.nan
    scale_perfly_curr[:,isreal] = mabe.compute_scale_perfly(Xcurr[...,isreal])

    Xcurr = Xcurr.reshape(sz+(Xcurr.shape[-2]*Xcurr.shape[-1],))
    ycurr = ycurr.reshape((ycurr.shape[0],ycurr.shape[-2]*ycurr.shape[-1]))
    isreal = mabe.get_real_flies(Xcurr.reshape((Xcurr.shape[0]*Xcurr.shape[1],Xcurr.shape[2])))
    Xcurr = Xcurr[:,:,isreal]
    ycurr = ycurr[:,isreal]
    flyidcurr = flyid0[isreal]
    idx = np.sort(np.random.choice(Xcurr.shape[2],nsample,replace=False))
    X[:,:,count*nsample:(count+1)*nsample] = Xcurr[:,:,idx]
    y[:,count*nsample:(count+1)*nsample] = ycurr[:,idx]

    minfly = np.min(flyidcurr[idx])
    maxfly = np.max(flyidcurr[idx])
    scale_perfly_curr = scale_perfly_curr[:,minfly:maxfly+1]
    flyid[count*nsample:(count+1)*nsample] = flyidcurr[idx]-minfly+flycount
    scale_perfly = np.concatenate((scale_perfly,scale_perfly_curr),axis=1)

    count += 1
    flycount += maxfly-minfly+1

  nscalefeats = len(mabe.scalenames)//2
  order=np.argsort(scale_perfly[mabe.scalenames.index('thorax_length'),:])
  colors = cm.Dark2(np.arange(nscalefeats)/nscalefeats)
  plt.clf()
  for i in range(nscalefeats):
    plt.errorbar(np.arange(len(order)),scale_perfly[i,order],scale_perfly[i+nscalefeats,order],fmt='none',color=colors[i,:-1],alpha=.1)

  for i in range(nscalefeats):
    plt.plot(np.arange(len(order)),scale_perfly[i,order],'.',color=colors[i,:-1],alpha=.5,label=mabe.scalenames[i])

  plt.legend()

  seqcurr = 1
  flycurr = 0
  id = all_dataset.seqids[seqcurr]
  Xcurr = all_dataset.X[id]
  isreal=mabe.get_real_flies(Xcurr.reshape((Xcurr.shape[0]*Xcurr.shape[1],)+Xcurr.shape[2:]))
  Xcurr = Xcurr[...,isreal]
  #scale_perfly_curr = compute_scale_perfly(Xcurr)
  #flyidcurr = np.tile(np.arange(Xcurr.shape[-1],dtype=int)[np.newaxis,:],(Xcurr.shape[-2],1))

  Xfeat = mabe.kp2feat(X,scale_perfly,flyid)
  Xfeatcurr,scale_perfly_curr,flyidcurr = mabe.kp2feat(Xcurr,return_scale=True)#,scale_perfly_curr,flyidcurr)

  plt.figure()
  plt.clf()
  yticks = np.zeros(len(mabe.posenames))
  maxsig = 4
  for i in range(len(mabe.posenames)):
    key = mabe.posenames[i]
    x = Xfeatcurr[mabe.posenames.index(key),...]
    dx = x
    #dx = np.diff(x,axis=0)
    if key == 'orientation' or len(re.findall('angle',key)) > 0:
      dx = mabe.modrange(dx,-np.pi,np.pi)

    mu = np.nanmean(dx.flatten())
    sig = np.nanstd(dx.flatten())
    dz = (dx-mu)/sig
    midy = (i+.5)*2*maxsig
    plt.plot(dz[:,flycurr]+midy,'-',label=f'delta {key}')
    yticks[i] = midy

  plt.gca().set_yticks(yticks)
  plt.gca().set_yticklabels(mabe.posenames)
  plt.gca().grid('on','major','both')

  plt.tight_layout()

  Xkp_reconstruct = mabe.feat2kp(Xfeat,scale_perfly,flyid)
  Xkp_reconstruct_curr = mabe.feat2kp(Xfeatcurr,scale_perfly_curr,flyidcurr)

  absheadangle = np.abs(Xfeatcurr[mabe.posenames.index('head_angle'),...])
  idx = np.unravel_index(np.argmax(absheadangle),absheadangle.shape)
  i = idx[0]
  fly = idx[1]

  # Xfeat1,scale1,flyid1 = mabe.kp2feat(Xcurr[:,:,i,fly],return_scale=True)
  # Xkpre1 = mabe.feat2kp(Xfeat1,scale1,flyid1)

  nfliescurr = Xcurr.shape[-1]
  naxc = int(np.ceil(np.sqrt(nfliescurr)))
  naxr = int(np.ceil(nfliescurr/naxc))
  fig,ax = plt.subplots(naxr,naxc)
  ax = ax.flatten()

  for fly in range(nfliescurr):
    all_dataset.plot_fly(Xcurr[:,:,i,fly],ax=ax[fly],color=[0,0,0],kpt_ms=12)
    all_dataset.plot_fly(Xkp_reconstruct_curr[:,:,i,fly],ax=ax[fly],color=[1,0,0],kpt_ms=12)
    ax[fly].axis('equal')


  plt.show()


def LoadFullData(loadfile=None):
  if loadfile is None:
    loadfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/fulldata20220118/Xusertrain.npz'
  with np.load(loadfile) as data:
    X = data['X']
    videoidx =  data['videoidx']
    ids = data['ids']
    frames = data['frames']
  return X,videoidx,ids,frames

if __name__ == "__main__":
  #PlotTrainLossDriver()
  #checkpointdir,loadfile = GetLatestCheckpoint()
  #loadfile = os.path.join(rootsavedir,'unet','PerCategoryBalanced20220124T202455','CP_latest_epoch108.pth')
  loadfile = None
  #PlotTrainLossDriver(loadfile=loadfile)
  #TrainDriver(loadfile=loadfile)
  #ClassCountDriver()
  #EvalDriver(loadfile=loadfile)
  #LoadFullData()
  #_ = CleanOldNets(deleteolddirs=True,deleteoldfiles=True)

  PoseRepDriver()
