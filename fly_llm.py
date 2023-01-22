import math
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm,animation,colors
import copy
import tqdm
from itertools import compress
import re
import MABeFlyUtils as mabe
import torch
import transformers
import warnings
import datetime


"""
data = load_raw_npz_data(inxfile,inyfile=None)
inxfile: npz file with pose data
inyfile: npz file with categories (optional). Default = None. 
Creates dict data with the following fields:
'X': T x maxnflies x d0 array of floats containing pose data for all flies and frames
'videoidx': T x 1 array of ints containing index of video pose is computed from
'ids': T x maxnflies array of ints containing fly id
'frames': T x 1 array of ints containing video frame number
'y': if inyfile is not None, this will be a T x maxnflies x ncategories binary matrix 
     indicating supervised behavior categories
"""
def load_raw_npz_data(infile):
  
  data = {}
  with np.load(infile) as data1:
    for key in data1:
      data[key] = data1[key]
    
  maxnflies = data['ids'].shape[1]
  # ids start at 1, make them start at 0
  data['ids'][data['ids']>=0] -= 1
  # starts of sequences, either because video changes or identity tracking issues
  # or because of filtering of training data
  isstart = (data['ids'][1:,:]!=data['ids'][:-1,:]) | \
    (data['frames'][1:,:] != (data['frames'][:-1,:]+1))
  isstart = np.concatenate((np.ones((1,maxnflies),dtype=bool),isstart),axis=0)
  
  data['isdata'] = data['ids'] >= 0
  data['isstart'] = isstart
  
  data['categories'] = list(data['categories'])
    
  return data

def filter_data_by_categories(data,categories):  
  iscategory = np.ones(data['y'].shape[1:],dtype=bool)
  for category in categories:
    if category == 'male':
      category = 'female'
      val = 0
    else:
      val = 1
    catidx = data['categories'].index(category)
    iscategory = iscategory & (data['y'][catidx,...]==val)
  data['isdata'] = data['isdata'] & iscategory

"""
y = interval_all(x,l)
Computes logical all over intervals of length l in the first dimension
y[i,j] is whether all entries in the l-length interval x[i:i+l,j] are true. 
x: input matrix of any shape. all will be computed over x[i:i+l,j,k]
outputs a matrix y of size (x.shape[0]-l,)+x.shape[1:]). 
"""
def interval_all(x,l):
  csx = np.concatenate((np.zeros((1,)+x.shape[1:],dtype=int),np.cumsum(x,axis=0)),axis=0)
  y = csx[l:-1,...]-csx[:-l-1,...] == l
  return y

"""
X = chunk_data(data,contextl,reparamfun)

"""
def chunk_data(data,contextl,reparamfun):
  
  contextlpad = contextl + 1
  
  # all frames for the main fly must have real data
  allisdata = interval_all(data['isdata'],contextlpad)
  isnotsplit = interval_all(data['isstart']==False,contextlpad-1)[1:,...]
  canstart = np.logical_and(allisdata,isnotsplit)

  # X is nkeypts x 2 x T x nflies
  nkeypoints = data['X'].shape[0]
  T = data['X'].shape[2]
  maxnflies = data['X'].shape[3]
  assert T > 2*contextlpad, 'Assumption that data has more frames than 2*(contextl+1) is incorrect, code will fail'
  
  # last possible start frame = T - contextl
  maxt0 = canstart.shape[0]-1
  # for computing the key, we will combine the frame and the id into one number for convenience
  # use a multiple of 10 so it is obvious when we look at the number
  Tmul = int(10**np.ceil(np.log10(T)))
  
  # X is a dict with chunked data
  X = []
  # loop through ids
  for flynum in tqdm.trange(maxnflies,desc='Fly'):
    # choose a first frame near the beginning, but offset a bit
    # first possible start
    canstartidx = np.nonzero(canstart[:,flynum])[0]
    if canstartidx.size == 0:
      continue

    mint0curr = canstartidx[0]
    # offset a bit
    t0 = mint0curr + np.random.randint(0,contextl,None)
    # find the next allowed frame
    if canstart[t0,flynum] == False:
      if not np.any(canstart[t0:,flynum]):
        continue
      t0 = np.nonzero(canstart[t0:,flynum])[0][0]+t0
    
    maxt0curr = canstartidx[-1]
    maxt1curr = maxt0curr+contextlpad-1
    for i in tqdm.trange(maxt1curr//contextl,desc='Frame'):
      if t0 > maxt0:
        break
      # this is guaranteed to be < T because of how 
      t1 = t0+contextlpad-1
      id = data['ids'][t0,flynum]
      xcurr = reparamfun(data['X'][...,t0:t1+1,:],id,flynum)
      xcurr['metadata'] = {'flynum': flynum, 'id': id, 't0': t0}
      xcurr['categories'] = data['y'][:,t0:t1+1,flynum].astype(np.float32)
      X.append(xcurr)
      if t0+contextl >= maxt0curr:
        break
      elif canstart[t0+contextl,flynum]:
        t0 = t0+contextl
      else:
        t0 = np.nonzero(canstart[t1+1:,flynum])[0]
        if t0 is None or t0.size == 0:
          break
        t0 = t0[0] + t1 + 1

  return X

def compute_scale_allflies(data):

  maxid = np.max(data['ids'])
  maxnflies = data['X'].shape[3]
  scale_perfly = None

  for flynum in range(maxnflies):

    idscurr = np.unique(data['ids'][data['ids'][:,flynum]>=0,flynum])
    for id in idscurr:
      idx = data['ids'][:,flynum] == id
      s = mabe.compute_scale_perfly(data['X'][...,idx,flynum])
      if scale_perfly is None:
        scale_perfly = np.zeros((s.size,maxid+1))
        scale_perfly[:] = np.nan
      else:
        assert(np.all(np.isnan(scale_perfly[:,id])))
      scale_perfly[:,id] = s.flatten()
      
  return scale_perfly

# compute vision features
#
# inputs:
# xeye_main: x-coordinate of main fly's position for vision. shape = (T).
# yeye_main: y-coordinate of main fly's position for vision. shape = (T).
# theta_main: orientation of main fly. shape = (T).
# xother: x-coordinate of 4 points on the other flies. shape = (4,T,nflies)
# yother: y-coordinate of 4 points on the other flies. shape = (4,T,nflies)
# params: dictionary of parameters with the following entries:
# params['n_oma']: number of bins representing visual scene. constant scalar. 
# params['mindist']: minimum distance from a fly to a chamber feature point. constant scalar.
# params['I']: y-coordinates in pixels of points along the chamber outline. shape = (n_oma).
# params['J']: x-coordinates in pixels of points along the chamber outline. shape = (n_oma).
#
# outputs:
# flyvision: appearance of other flies to input fly. shape = (n_oma).
# chambervision: appearance of arena to input fly. shape = (n_oma).
def compute_sensory(xeye_main,yeye_main,theta_main,
                    xlegtip_main,ylegtip_main,
                    xother,yother,params):

  # increase dimensions if only one frame input
  if xother.ndim < 3:
    xother = xother[:,None]

  npts = xother.shape[0]
  nflies = xother.shape[1]
  T = xother.shape[2]
  nlegtips = xlegtip_main.shape[0]
  
  yother = np.reshape(yother,(npts,nflies,T))
  xeye_main = np.reshape(xeye_main,(1,1,T))
  yeye_main = np.reshape(yeye_main,(1,1,T))
  theta_main = np.reshape(theta_main,(1,1,T))
  
  # don't deal with missing data :)    
  assert(np.any(np.isnan(xeye_main))==False)
  assert(np.any(np.isnan(yeye_main))==False)
  assert(np.any(np.isnan(theta_main))==False)
  
  # vision bin size
  step = 2.*np.pi/params['n_oma']

  # compute other flies view

  # convert to this fly's coord system
  dx = xother-xeye_main
  dy = yother-yeye_main
  
  # distance
  dist = np.sqrt(dx**2+dy**2)
  
  # angle in the original coordinate system
  angle0 = np.arctan2(dy,dx)  
  
  # subtract off angle of main fly
  angle = angle0 - theta_main
  angle = mabe.modrange(angle,-np.pi,np.pi)

  # which other flies pass beyond the -pi to pi border
  isbackpos = angle > np.pi/2
  isbackneg = angle < -np.pi/2
  isfront = np.abs(angle) <= np.pi/2
  idxmod = np.any(isbackpos,axis=0) & np.any(isbackneg,axis=0) & (np.any(isfront,axis=0)==False)

  # bin - npts x nflies x T
  b_all = np.floor((angle+np.pi)/step)
  
  # bin range
  # shape: nflies x T
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    minb = np.nanmin(b_all,axis=0)
    maxb = np.nanmax(b_all,axis=0)
    mind = np.nanmin(dist,axis=0) 
    
  # n_oma x 1 x 1
  tmpbins = np.arange(params['n_oma'])[:,None,None]
  
  # n_oma x nflies x T
  mindrep = np.tile(mind[None,...],(params['n_oma'],1,1))
  mask = (tmpbins >= minb[None,...]) & (tmpbins <= maxb[None,...])
  
  if np.any(idxmod):
    # this is complicated!! 
    # find the max bin for negative angles
    # and the min bin for positive angles
    # store them in min and max for consistency with non-wraparound
    isbackpos1 = isbackpos[:,idxmod]
    isbackneg1 = isbackneg[:,idxmod]
    bmodneg = b_all[:,idxmod]
    bmodneg[isbackpos1] = np.nan
    minbmod = np.nanmax(bmodneg,axis=0)
    bmodpos = b_all[:,idxmod]
    bmodpos[isbackneg1] = np.nan
    maxbmod = np.nanmin(bmodpos,axis=0)
    mask[:,idxmod] = (tmpbins[...,0] >= maxbmod[None,:]) | (tmpbins[...,0] <= minbmod[None,:])
  
  otherflies_vision = np.nanmin(np.where(mask,mindrep,np.inf),axis=1,initial=np.inf)
  
  otherflies_vision = 1. - np.minimum(1., params['otherflies_vision_mult'] * otherflies_vision**params['otherflies_vision_exp'])

  if False:
    t = 249
    rplot = 2*params['outer_arena_radius']
    plt.figure()
    ax = plt.subplot(1,3,1)
    hother = ax.plot(xother[:,:,t],yother[:,:,t],'.-')
    ax.set_aspect('equal')
    #ax.plot(X[:,0,0,flynum],X[:,1,0,flynum],'k.')
    ax.plot(xeye_main[0,0,t],yeye_main[0,0,t],'r.')
    ax.plot([xeye_main[0,0,t],xeye_main[0,0,t]+rplot*np.cos(theta_main[0,0,t])],
            [yeye_main[0,0,t],yeye_main[0,0,t]+rplot*np.sin(theta_main[0,0,t])],'r--')
    for tmpfly in range(nflies):
      ax.plot(xeye_main[0,0,t]+np.c_[np.zeros((npts,1)),np.cos(angle0[:,tmpfly,t])*rplot].T,
              yeye_main[0,0,t]+np.c_[np.zeros((npts,1)),np.sin(angle0[:,tmpfly,t])*rplot].T,
              color=hother[tmpfly].get_color(),alpha=.5)

    ax = plt.subplot(1,3,2)  
    for tmpfly in range(nflies):
      ax.plot(np.c_[np.zeros((npts,1)),np.cos(angle[:,tmpfly,t])].T,
              np.c_[np.zeros((npts,1)),np.sin(angle[:,tmpfly,t])].T,
              color=hother[tmpfly].get_color(),alpha=.5)
    ax.plot(0,0,'r.')
    ax.plot([0,1],[0,0],'r--')
    ax.set_aspect('equal')

    ax = plt.subplot(1,3,3)
    for tmpfly in range(nflies):
      ax.plot(b_all[:,tmpfly,t],dist[:,tmpfly,t],'o',color=hother[tmpfly].get_color())
    ax.set_xlim([-.5,params['n_oma']-.5])
    ax.set_xlabel('bin')
    ax.set_ylabel('dist')

    tmpvision = np.minimum(50,otherflies_vision[:,t])
    ax.plot(tmpvision,'k-')

  # distance from center of arena
  # center of arena is assumed to be [0,0]
  distleg = np.sqrt( xlegtip_main**2. + ylegtip_main**2 )

  # height of chamber 
  wall_touch = np.zeros(distleg.shape)
  wall_touch[:] = params['arena_height']
  wall_touch = np.minimum(params['arena_height'],np.maximum(0.,params['arena_height'] - (distleg-params['inner_arena_radius'])*params['arena_height']/(params['outer_arena_radius']-params['inner_arena_radius'])))
  wall_touch[distleg >= params['outer_arena_radius']] = 0.
  
  if False:
    plt.figure()
    plt.clf()
    t = 0
    ax = plt.subplot(1,2,1)
    ax.plot(xlegtip_main.flatten(),ylegtip_main.flatten(),'k.')
    theta_arena = np.linspace(-np.pi,np.pi,100)
    ax.plot(np.cos(theta_arena)*params['inner_arena_radius'],np.sin(theta_arena)*params['inner_arena_radius'],'-')
    ax.plot(np.cos(theta_arena)*params['outer_arena_radius'],np.sin(theta_arena)*params['outer_arena_radius'],'-')
    hpts = []
    for pti in range(nlegtips):
      hpts.append(ax.plot(xlegtip_main[pti,t],ylegtip_main[pti,t],'o')[0])
    ax.set_aspect('equal')
    ax = plt.subplot(1,2,2)
    ax.plot(distleg.flatten(),wall_touch.flatten(),'k.')
    ax.plot([0,params['inner_arena_radius'],params['outer_arena_radius']],
            [params['arena_height'],params['arena_height'],0],'-')
    for pti in range(nlegtips):
      ax.plot(distleg[pti,t],wall_touch[pti,t],'o',color=hpts[pti].get_color())
    ax.set_aspect('equal')

  # to do: add something related to whether the fly is touching another fly
  
  return (otherflies_vision, wall_touch)


featorigin = [mabe.posenames.index('thorax_front_x'),mabe.posenames.index('thorax_front_y')]
feattheta = mabe.posenames.index('orientation')
featglobal = featorigin + [feattheta,]
featrelative = np.ones(len(mabe.posenames),dtype=bool)
featrelative[featglobal] = False
nrelative = np.count_nonzero(featrelative)
featangle = np.array([re.search('angle$',s) is not None for s in mabe.posenames])
featangle[feattheta] = True

kpother = [mabe.keypointnames.index('antennae_midpoint'),
            mabe.keypointnames.index('tip_abdomen'),
            mabe.keypointnames.index('left_middle_femur_base'),
            mabe.keypointnames.index('right_middle_femur_base'),
            ]
kpeye = mabe.keypointnames.index('antennae_midpoint')
kplegtip = [mabe.keypointnames.index('right_front_leg_tip'),
            mabe.keypointnames.index('right_middle_leg_tip'),
            mabe.keypointnames.index('right_back_leg_tip'),
            mabe.keypointnames.index('left_back_leg_tip'),
            mabe.keypointnames.index('left_middle_leg_tip'),
            mabe.keypointnames.index('left_front_leg_tip')]
nlegtips = len(kplegtip)

def split_features(X):
  res = {}
  res['pose'] = X[...,:nrelative]
  res['otherflies_vision'] = X[...,nrelative:-nlegtips]
  res['wall_touch'] = X[...,-nlegtips:]
  return res

def combine_relative_global(Xrelative,Xglobal):
  X = np.concatenate((Xglobal,Xrelative),axis=-1)
  return X

def compute_features(X,id,flynum,scale_perfly,sensory_params,smush=True,outtype=None):
  
  # convert to relative locations of body parts
  Xfeat = mabe.kp2feat(X[...,flynum],scale_perfly[:,id])
  Xfeat = Xfeat[...,0]

  # compute sensory information

  # other flies positions
  idxother = np.ones(X.shape[-1],dtype=bool)
  idxother[flynum] = False
  Xother = X[:,:,:,idxother]
  
  xeye_main = X[kpeye,0,:,flynum]
  yeye_main = X[kpeye,1,:,flynum]
  xlegtip_main = X[kplegtip,0,:,flynum]
  ylegtip_main = X[kplegtip,1,:,flynum]
  xother = Xother[kpother,0,...].transpose((0,2,1))
  yother = Xother[kpother,1,...].transpose((0,2,1))
  theta_main = Xfeat[feattheta,...]+np.pi/2
  
  otherflies_vision,wall_touch = \
    compute_sensory(xeye_main,yeye_main,theta_main,
                    xlegtip_main,ylegtip_main,
                    xother,yother,sensory_params)
    
  movement = Xfeat[:,1:]-Xfeat[:,:-1]
  movement[featangle,...] = mabe.modrange(movement[featangle,...],-np.pi,np.pi)

  res = {}
  res['input'] = np.r_[Xfeat[featrelative,:-1],wall_touch[:,:-1],otherflies_vision[:,:-1]].T
  res['labels'] = movement.T
  res['init'] = Xfeat[featrelative==False,0]
  res['scale'] = scale_perfly[:,id]
  
  if not smush:
    res['global'] = Xfeat[featrelative==False,:-1]
    res['relative'] = Xfeat[featrelative,:-1]
    res['wall_touch'] = wall_touch[:,:-1]
    res['otherflies_vision'] = otherflies_vision[:,:-1]
    
  # debug_plot_compute_features(X,porigin,theta,Xother,Xnother)
    
  if outtype is not None:
    res = {key: val.astype(outtype) for key,val in res.items()}
  return res
    
def debug_plot_compute_features(X,porigin,theta,Xother,Xnother):
  
  t = 0
  rplot = 5.
  plt.clf()
  ax = plt.subplot(1,2,1)
  mabe.plot_flies(X[:,:,t,:],kptidx=mabe.keypointidx,skelidx=mabe.skeleton_edges,
                  ax=ax,textlabels='fly',colors=np.zeros((X.shape[-1],3)))
  ax.plot(porigin[0,t],porigin[1,t],'rx',linewidth=2)
  ax.plot([porigin[0,t,0],porigin[0,t,0]+np.cos(theta[t,0])*rplot],
          [porigin[1,t,0],porigin[1,t,0]+np.sin(theta[t,0])*rplot],'r-')
  ax.plot(Xother[kpother,0,t,:],Xother[kpother,1,t,:],'o')
  ax.set_aspect('equal')
  
  ax = plt.subplot(1,2,2)
  ax.plot(0,0,'rx')
  ax.plot([0,np.cos(0.)*rplot],[0,np.sin(0.)*rplot],'r-')
  ax.plot(Xnother[:,0,t,:],Xnother[:,1,t,:],'o')
  ax.set_aspect('equal')

def apply_mask(x,mask):
  # mask with zeros
  x[mask,:] = 0.
  
  return x

def unzscore(x,mu,sig):
  return x*sig + mu


class FlyMLMDataset(torch.utils.data.Dataset):
  def __init__(self,data,max_mask_length=None,pmask=None,masktype='block'):
    self.data = data
    self.max_mask_length = max_mask_length
    self.pmask = pmask
    self.masktype = masktype
    
    self.mu_input = None
    self.sig_input = None
    self.mu_labels = None
    self.sig_labels = None
    
  def zscore(self):
    
    def zscore_helper(data,f):
      mu = 0.
      sig = 0.
      n = 0.
      for example in self.data:
        # input is T x dfeat
        n += np.sum(np.isnan(example[f]) == False,axis=0)
        mu += np.nansum(example[f],axis=0)
        sig += np.nansum(example[f]**2.,axis=0)
      mu = mu / n
      sig = np.sqrt(sig/n - mu**2.)
      assert(np.any(np.isnan(mu))==False)
      assert(np.any(np.isnan(sig))==False)

      for example in self.data:    
        example[f] = ((example[f]-mu)/sig).astype(np.float32)
        
      return mu,sig
    
    self.mu_input,self.sig_input = zscore_helper(self.data,'input')
    self.mu_labels,self.sig_labels = zscore_helper(self.data,'labels')
      
  def maskblock(self,x):
    # choose a mask length
    l = np.random.randint(1,self.max_mask_length)
    
    # choose mask start
    t0 = np.random.randint(0,x.shape[0]-l)
    t1 = t0+l
    
    # create mask
    mask = torch.zeros(x.shape[0],dtype=bool)
    mask[t0:t1] = True
        
    x = apply_mask(x,mask)
    
    return x,mask
  
  def masklast(self,x):
    t0 = x.shape[0]-1
    t1 = t0+1
    mask = torch.zeros(x.shape[0],dtype=bool)
    mask[t0:t1] = True
    x = apply_mask(x,mask)
    return x,mask
  
  def maskind(self,x):
    mask = torch.rand(x.shape[0])<=self.pmask
    x = apply_mask(x,mask)
    return x,mask
  
  def set_masktype(self,masktype):
    self.masktype = masktype
    
  def __getitem__(self,idx):
    input = torch.as_tensor(self.data[idx]['input'].copy())
    # to do: do we need to copy labels? we don't destroy it
    labels = torch.as_tensor(self.data[idx]['labels'].copy())
    init = torch.as_tensor(self.data[idx]['init'].copy())
    scale = torch.as_tensor(self.data[idx]['scale'].copy())
    categories = torch.as_tensor(self.data[idx]['categories'].copy())
    if self.masktype == 'block':
      input,mask = self.maskblock(input)
    elif self.masktype == 'ind':
      input,mask = self.maskind(input)
    elif self.masktype == 'last':
      input,mask = self.masklast(input)
    return {'input': input, 'labels': labels, 'mask': mask, 
            'init': init, 'scale': scale, 'categories': categories,
            'metadata': self.data[idx]['metadata'].copy()}
  
  def __len__(self):
    return len(self.data)
    
  def getitem_raw_np(self,idx):
    if self.mu_input is None:
      return self.data[idx]
    res['input'] = unzscore(self.data[idx]['input'],self.mu_input,self.sig_input)
    res['labels'] = unzscore(self.data[idx]['labels'],self.mu_labels,self.sig_labels)
    res['init'] = self.data[idx]['init'].copy()
    res['scale'] = self.data[idx]['scale'].copy()
    return res
  
  def get_Xfeat(self,feat0,global0,movements):
    if self.mu_input is not None:
      feat0 = unzscore(feat0,self.mu_input,self.sig_input)
      movements = unzscore(movements,self.mu_labels,self.sig_labels)
      
    feat0 = split_features(feat0)
    feat0 = combine_relative_global(feat0['pose'],global0)
    Xfeat = np.r_[feat0[None,:],movements]
    Xfeat = np.cumsum(Xfeat,axis=0)
    
    return Xfeat
  
  def get_Xkp(self,feat0,global0,movements,scale):
    Xfeat = self.get_Xfeat(feat0,global0,movements)
    Xkp = mabe.feat2kp(Xfeat.T[...,None],scale[...,None])
    return Xkp
    
  
def masked_criterion(tgt,pred,mask):
  err = torch.sum((tgt[mask,:]-pred[mask,:])**2.)
  return err


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(torch.nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()

    # during training, randomly zero some of the inputs with probability p=dropout
    self.dropout = torch.nn.Dropout(p=dropout)

    # compute sine and cosine waves at different frequencies
    # pe[0,:,i] will have a different value for each word (or whatever)
    # will be sines for even i, cosines for odd i,
    # exponentially decreasing frequencies with i
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
    pe = torch.zeros(1,max_len,d_model)
    pe[0,:,0::2] = torch.sin(position * div_term)
    pe[0,:,1::2] = torch.cos(position * div_term)

    # buffers will be saved with model parameters, but are not model parameters
    self.register_buffer('pe', pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
      x: Tensor, shape [batch_size, seq_len, embedding_dim]
    """

    # add positional encoding
    x = x + self.pe[:,:x.size(1),:]

    # zero out a randomly selected subset of entries
    return self.dropout(x)

  
class TransformerModel(torch.nn.Module):

  def __init__(self, d_input: int, d_output: int,
               d_model: int = 2048, nhead: int = 8, d_hid: int = 512,
               nlayers: int = 12, dropout: float = 0.1):
    super().__init__()
    self.model_type = 'Transformer'

    # frequency-based representation of word position with dropout
    self.pos_encoder = PositionalEncoding(d_model,dropout)

    # create self-attention + feedforward network module
    # d_model: number of input features
    # nhead: number of heads in the multiheadattention models
    # dhid: dimension of the feedforward network model
    # dropout: dropout value
    encoder_layers = torch.nn.TransformerEncoderLayer(d_model,nhead,d_hid,dropout,batch_first=True)

    # stack of nlayers self-attention + feedforward layers
    # nlayers: number of sub-encoder layers in the encoder
    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers,nlayers)

    # encoder and decoder are currently not tied together, but maybe they should be? 
    # fully-connected layer from input size to d_model
    self.encoder = torch.nn.Linear(d_input,d_model)

    # fully-connected layer from d_model to input size
    self.decoder = torch.nn.Linear(d_model,d_output)

    # store hyperparameters
    self.d_model = d_model

    self.init_weights()

  def init_weights(self) -> None:
    pass

  def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
      src: Tensor, shape [seq_len,batch_size,dinput]
      src_mask: Tensor, shape [seq_len,seq_len]
    Returns:
      output Tensor of shape [seq_len, batch_size, ntoken]
    """

    # project input into d_model space, multiple by sqrt(d_model) for reasons?
    src = self.encoder(src) * math.sqrt(self.d_model)

    # add in the positional encoding of where in the sentence the words occur
    # it is weird to me that these are added, but I guess it would be almost
    # the same to have these be combined in a single linear layer
    src = self.pos_encoder(src)

    # main transformer layers
    output = self.transformer_encoder(src,src_mask)

    # project back to d_input space
    output = self.decoder(output)

    return output

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
  """
  Generates an upper-triangular matrix of -inf, with zeros on and below the diagonal.
  This is used to restrict attention to the past when predicting future words. Only
  used for causal models, not masked models.
  """
  return torch.triu(torch.ones(sz,sz) * float('-inf'), diagonal=1)

def generate_square_full_mask(sz: int) -> torch.Tensor:
  """
  Generates an zero matrix. All words allowed.
  """
  return torch.zeros(sz,sz)

def save_model(savefile,model,lr_optimizer=None,scheduler=None,loss=None):
  tosave = {'model':model.state_dict()}
  if lr_optimizer is not None:
    tosave['lr_optimizer'] = lr_optimizer.state_dict()
  if scheduler is not None:
    tosave['scheduler'] = scheduler.state_dict()
  if loss is not None:
    tosave['loss'] = loss
  torch.save(tosave,savefile)
  return

def load_model(loadfile,model,lr_optimizer=None,scheduler=None):
  print(f'Loading model from file {loadfile}...')
  state = torch.load(loadfile, map_location=device)
  if model is not None:
    model.load_state_dict(state['model'])
  if lr_optimizer is not None and ('lr_optimizer' in state):
    lr_optimizer.load_state_dict(state['lr_optimizer'])
  if scheduler is not None and ('scheduler' in state):
    scheduler.load_state_dict(state['scheduler'])
  loss = None
  if 'loss' in state:
    loss = state['loss']
  return loss

# location of data
datadir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022'
infile = os.path.join(datadir,'usertrain.npz')
categories = ['71G01','male',]
#inchunkfile = os.path.join(datadir,'chunk_usertrain.npz')
inchunkfile = None
#savechunkfile = os.path.join(datadir,f'chunk_usertrain_{"_".join(categories)}.npz')
savechunkfile = None
savedir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets'
loadmodelfile = os.path.join(savedir,'flymlm_71G01_male_epoch100_202301012534.pth')

# parameters

narena = 2**10
theta_arena = np.linspace(-np.pi,np.pi,narena+1)[:-1]
SENSORY_PARAMS = {
  'n_oma': 72,
  'inner_arena_radius': 17.5, # in mm
  'outer_arena_radius': mabe.ARENA_RADIUS_MM,
  'arena_height': 3.5,
  'otherflies_vision_exp': .6,
}
SENSORY_PARAMS['otherflies_vision_mult'] = 1./((2.*mabe.ARENA_RADIUS_MM)**SENSORY_PARAMS['otherflies_vision_exp'])

# sequence length per training example
CONTEXTL = 512
# masking method
MASKTYPE = 'ind'
# how long to mask out at a time (maximum), masktype = 'block'
MAX_MASK_LENGTH = 64
# probability of masking, masktype = 'ind'
PMASK = .15
# batch size
BATCH_SIZE = 32
# number of training epochs
NUM_TRAIN_EPOCHS = 100
# gradient descent parameters
OPTIMIZER_ARGS = {'lr': 5e-5, 'betas': (.9,.999), 'eps': 1e-8}
MAX_GRAD_NORM = 1.
# architecture arguments
MODEL_ARGS = {'d_model': 2048, 'nhead': 8, 'd_hid': 512, 'nlayers': 6, 'dropout': .1}

NUMPY_SEED = 123
TORCH_SEED = 456

np.random.seed(NUMPY_SEED)
torch.manual_seed(TORCH_SEED)
device = torch.device('cuda')

# load in data
data = load_raw_npz_data(infile)

# filter out data
if categories is not None and len(categories) > 0:
  filter_data_by_categories(data,categories)

# compute scale parameters
scale_perfly = compute_scale_allflies(data)

# throw out data that is missing scale information - not so many frames
idsremove = np.nonzero(np.any(np.isnan(scale_perfly),axis=0))[0]
data['isdata'][np.isin(data['ids'],idsremove)] = False

# function for computing features
reparamfun = lambda x,id,flynum: compute_features(x,id,flynum,scale_perfly,SENSORY_PARAMS,outtype=np.float32)

# group data and compute features
if inchunkfile is None or not os.path.exists(inchunkfile):
  print('Chunking data...')
  X = chunk_data(data,CONTEXTL,reparamfun)
  print('Done.')
  if savechunkfile is not None:
    np.savez(savechunkfile,X=X)
else:
  print('Loading chunked data')
  res = np.load(inchunkfile,allow_pickle=True)
  X = list(res['X'])
  print('Done.')

train_dataset = FlyMLMDataset(X,max_mask_length=MAX_MASK_LENGTH,pmask=PMASK,masktype=MASKTYPE)
train_dataset.zscore()

d_feat = train_dataset[0]['input'].shape[-1]
d_output = train_dataset[0]['labels'].shape[-1]

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               pin_memory=True,
                                              )
ntrain = len(train_dataloader)

# spot check that we can get poses from examples
example = next(iter(train_dataloader))

ntsplot = 5
nsamplesplot = 3
fig,ax = plt.subplots(nsamplesplot,ntsplot,sharex='row',sharey='row')
tsplot = np.round(np.linspace(0,CONTEXTL,ntsplot)).astype(int)
samplesplot = np.round(np.linspace(0,BATCH_SIZE-1,nsamplesplot)).astype(int)
h = {'kpt0': [], 'kpt1': [], 'edge0': [], 'edge1': []}
for i in range(nsamplesplot):
  iplot = samplesplot[i]
  Xkp = train_dataset.get_Xkp(example['input'][iplot,0,...].numpy(),
                              example['init'][iplot,...].numpy(),
                              example['labels'][iplot,...].numpy(),
                              example['scale'][iplot,...].numpy())
  t0 = example['metadata']['t0'][iplot].item()
  flynum = example['metadata']['flynum'][iplot].item()
  for key in h.keys():
    h[key].append([None,]*ntsplot)
  for j in range(ntsplot):
    tplot = tsplot[j]
    ax[i,j].set_title(f't = {tplot}, sample = {iplot}')
    h['kpt0'][i][j],h['edge0'][i][j],_,_,_ = mabe.plot_fly(Xkp[:,:,tplot,0],kptidx=mabe.keypointidx,skelidx=mabe.skeleton_edges,skel_lw=2,ax=ax[i,j],color=[0,0,0])
    h['kpt1'][i][j],h['edge1'][i][j],_,_,_ = mabe.plot_fly(data['X'][:,:,t0+tplot,flynum],kptidx=mabe.keypointidx,skelidx=mabe.skeleton_edges,ax=ax[i,j],skel_lw=1,color=[0,1,1])
    
    ax[i,j].set_aspect('equal')
plt.ion()
plt.show()
plt.pause(.001)

model = TransformerModel(d_feat,d_output,**MODEL_ARGS).to(device)

num_training_steps = NUM_TRAIN_EPOCHS * ntrain
optimizer = transformers.optimization.AdamW(model.parameters(),**OPTIMIZER_ARGS)
lr_scheduler = transformers.get_scheduler('linear',optimizer,num_warmup_steps=0,
                                          num_training_steps=num_training_steps)

train_loss_epoch = torch.zeros(NUM_TRAIN_EPOCHS)
train_loss_iter = torch.zeros(num_training_steps)

progress_bar = tqdm.tqdm(range(num_training_steps))

train_src_mask = generate_square_full_mask(CONTEXTL).to(device)

if loadmodelfile is None:
  for epoch in range(NUM_TRAIN_EPOCHS):
    
    model.train()
    tr_loss = torch.tensor(0.0).to(device)

    for step, example in enumerate(train_dataloader):
      
      pred = model(example['input'].to(device=device),train_src_mask)
      loss = masked_criterion(example['labels'].to(device=device),pred,
                              example['mask'].to(device))
      loss.backward()

      if step % 10 == 0:
        for i in range(nsamplesplot):
          iplot = samplesplot[i]
          maskidx = example['mask'][iplot].nonzero()
          t0plot = maskidx[0].item()
          t1plot = maskidx[-1].item()
          Xkp = train_dataset.get_Xkp(example['input'][iplot,0,...].numpy(),
                                      example['init'][iplot,...].numpy(),
                                      example['labels'][iplot,...].numpy(),
                                      example['scale'][iplot,...].numpy())
          predkp = train_dataset.get_Xkp(example['input'][iplot,0,...].numpy(),
                                        example['init'][iplot,...].numpy(),
                                        pred[iplot,...].detach().cpu().numpy(),
                                        example['scale'][iplot,...].numpy())
          for j in range(ntsplot):
            tplot = tsplot[j]
            mabe.plot_fly(Xkp[:,:,tplot,0],
                          kptidx=mabe.keypointidx,skelidx=mabe.skeleton_edges,
                          hkpt=h['kpt0'][i][j],hedge=h['edge0'][i][j],ax=ax[i,j])
            mabe.plot_fly(predkp[:,:,tplot,0],
                          kptidx=mabe.keypointidx,skelidx=mabe.skeleton_edges,
                          hkpt=h['kpt1'][i][j],hedge=h['edge1'][i][j],ax=ax[i,j])
            ax[i,j].relim()
            ax[i,j].autoscale()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(.001)

      tr_loss_step = loss.detach()
      tr_loss += tr_loss_step

      # gradient clipping
      torch.nn.utils.clip_grad_norm_(model.parameters(),MAX_GRAD_NORM)
      optimizer.step()
      lr_scheduler.step()
      model.zero_grad()
      progress_bar.set_postfix({'train loss': tr_loss.item()/(step+1)})
      progress_bar.update(1)

    train_loss_epoch[epoch] = tr_loss.item() / ntrain

  savetime = datetime.datetime.now()
  savetime = savetime.strftime('%Y%m%H%M%S')
  savefile = os.path.join(savedir,f'flymlm_{"_".join(categories)}_epoch{epoch+1}_{savetime}.pth')


  save_model(savefile,model,lr_optimizer=optimizer,scheduler=lr_scheduler,loss=train_loss_epoch)

  print('Done training')
else:
  train_loss_epoch = load_model(loadmodelfile,model,lr_optimizer=optimizer,scheduler=lr_scheduler)
    
example = next(iter(train_dataloader))
model.eval()

with torch.no_grad():
  pred = model(example['input'].to(device=device),train_src_mask)

iplot = 1
Xkp_true = train_dataset.get_Xkp(example['input'][iplot,0,...].numpy(),
                                 example['init'][iplot,...].numpy(),
                                 example['labels'][iplot,...].numpy(),
                                 example['scale'][iplot,...].numpy())
Xkp_pred = train_dataset.get_Xkp(example['input'][iplot,0,...].numpy(),
                                 example['init'][iplot,...].numpy(),
                                 pred[iplot,...].cpu().numpy(),
                                 example['scale'][iplot,...].numpy())

fig1,ax1 = plt.subplots(1,2,sharex='all',sharey='all')

mabe.plot_fly(Xkp_true[:,:,0,0],kptidx=mabe.keypointidx,skelidx=mabe.skeleton_edges,skel_lw=1,ax=ax1[0],color=[0,1,1])
mabe.plot_fly(Xkp_pred[:,:,0,0],kptidx=mabe.keypointidx,skelidx=mabe.skeleton_edges,ax=ax1[1],skel_lw=1,color=[0,1,1])
hkpt0,hedge0,_,_,_ = mabe.plot_fly(Xkp_true[:,:,0,0],kptidx=mabe.keypointidx,skelidx=mabe.skeleton_edges,skel_lw=2,ax=ax1[0],color=[0,0,0])
hkpt1,hedge1,_,_,_ = mabe.plot_fly(Xkp_pred[:,:,0,0],kptidx=mabe.keypointidx,skelidx=mabe.skeleton_edges,ax=ax1[1],skel_lw=2,color=[0,0,0])

minx_true = np.min(Xkp_true[:,0,...])
maxx_true = np.max(Xkp_true[:,0,...])
miny_true = np.min(Xkp_true[:,1,...])
maxy_true = np.max(Xkp_true[:,1,...])

minx_pred = np.min(Xkp_pred[:,0,...])
maxx_pred = np.max(Xkp_pred[:,0,...])
miny_pred = np.min(Xkp_pred[:,1,...])
maxy_pred = np.max(Xkp_pred[:,1,...])

minx = np.minimum(minx_pred,minx_true)
maxx = np.maximum(maxx_pred,maxx_true)
miny = np.minimum(miny_pred,miny_true)
maxy = np.maximum(maxy_pred,maxy_true)
ax1[0].set_xlim([minx,maxx])
ax1[0].set_ylim([miny,maxy])
ax1[0].set_aspect('equal')
ax1[1].set_aspect('equal')

for t in range(Xkp_true.shape[2]):
  ax1[0].set_title(f'True, t = {t}')
  ax1[1].set_title(f'Pred, t = {t}')
  mabe.plot_fly(Xkp_true[:,:,t,0],kptidx=mabe.keypointidx,skelidx=mabe.skeleton_edges,hkpt=hkpt0,hedge=hedge0,ax=ax1[0])
  mabe.plot_fly(Xkp_pred[:,:,t,0],kptidx=mabe.keypointidx,skelidx=mabe.skeleton_edges,hkpt=hkpt1,hedge=hedge1,ax=ax1[1])
  plt.pause(.001)