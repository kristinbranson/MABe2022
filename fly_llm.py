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
import sklearn.preprocessing

SENSORY_PARAMS = {
  'n_oma': 72,
  'inner_arena_radius': 17.5, # in mm
  'outer_arena_radius': mabe.ARENA_RADIUS_MM,
  'arena_height': 3.5,
  'otherflies_vision_exp': .6,
}
SENSORY_PARAMS['otherflies_vision_mult'] = 1./((2.*mabe.ARENA_RADIUS_MM)**SENSORY_PARAMS['otherflies_vision_exp'])

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
    #maxt1curr = maxt0curr+contextlpad-1
    ndata = np.count_nonzero(data['isdata'][:,flynum])
    maxintervals = ndata//contextl+1
    for i in tqdm.trange(maxintervals,desc='Interval'):
      if t0 > maxt0:
        break
      # this is guaranteed to be < T
      t1 = t0+contextlpad-1
      id = data['ids'][t0,flynum]
      xcurr = reparamfun(data['X'][...,t0:t1+1,:],id,flynum)
      xcurr['metadata'] = {'flynum': flynum, 'id': id, 't0': t0, 'videoidx': data['videoidx'][t0], 'frame0': data['frames'][t0]}
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

def debug_plot_otherflies_vision(t,xother,yother,xeye_main,yeye_main,theta_main,
                                 angle0,angle,dist,b_all,otherflies_vision,params):  
  npts = xother.shape[0]
  nflies = xother.shape[1]
  
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
  
def debug_plot_wall_touch(t,xlegtip_main,ylegtip_main,distleg,wall_touch,params):
  plt.figure()
  plt.clf()
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

  # t = 249
  # debug_plot_otherflies_vision(t,xother,yother,xeye_main,yeye_main,theta_main,
  #                                 angle0,angle,dist,b_all,otherflies_vision,params)

  # distance from center of arena
  # center of arena is assumed to be [0,0]
  distleg = np.sqrt( xlegtip_main**2. + ylegtip_main**2 )

  # height of chamber 
  wall_touch = np.zeros(distleg.shape)
  wall_touch[:] = params['arena_height']
  wall_touch = np.minimum(params['arena_height'],np.maximum(0.,params['arena_height'] - (distleg-params['inner_arena_radius'])*params['arena_height']/(params['outer_arena_radius']-params['inner_arena_radius'])))
  wall_touch[distleg >= params['outer_arena_radius']] = 0.
  
  # t = 0
  # debug_plot_wall_touch(t,xlegtip_main,ylegtip_main,distleg,wall_touch,params)

  # to do: add something related to whether the fly is touching another fly
  
  return (otherflies_vision, wall_touch)


featorigin = [mabe.posenames.index('thorax_front_x'),mabe.posenames.index('thorax_front_y')]
feattheta = mabe.posenames.index('orientation')
featglobal = featorigin + [feattheta,]
featrelative = np.ones(len(mabe.posenames),dtype=bool)
featrelative[featglobal] = False
nrelative = np.count_nonzero(featrelative)
nfeatures = len(mabe.posenames)
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

def split_features(X,simplify=None):
  res = {}
  res['pose'] = X[...,:nrelative]
  if simplify == 'no_sensory':
    res['otherflies_vision'] = None
    res['wall_touch'] = None
  else:
    i0 = nrelative
    i1 = nrelative+SENSORY_PARAMS['n_oma']
    res['otherflies_vision'] = X[...,i0:i1]
    i0 = i1
    i1 = i0 + nlegtips
    res['wall_touch'] = X[...,i0:i1]
  return res

def combine_relative_global(Xrelative,Xglobal):
  X = np.concatenate((Xglobal,Xrelative),axis=-1)
  return X

def compute_global(Xkp):
   _,fthorax,thorax_theta = mabe.body_centric_kp(Xkp)
   return fthorax,thorax_theta

def compute_features(X,id,flynum,scale_perfly,sensory_params,smush=True,outtype=None,
                     simplify_out=None,simplify_in=None):
  
  res = {}
  
  # convert to relative locations of body parts
  Xfeat = mabe.kp2feat(X[...,flynum],scale_perfly[:,id])
  Xfeat = Xfeat[...,0]
  
  # compute global coordinates relative to previous frame
  Xorigin = Xfeat[featorigin,...]
  Xtheta = Xfeat[feattheta,...]
  dXoriginrel = mabe.rotate_2d_points((Xorigin[:,1:]-Xorigin[:,:-1]).T,Xtheta[:-1]).T
  forward_vel = dXoriginrel[1,:]
  sideways_vel = dXoriginrel[0,:]

  # compute sensory information

  if simplify_in == 'no_sensory':
    input = Xfeat[featrelative,:].T
    res['input'] = input[:-1,:]
  else:
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
    input = np.r_[Xfeat[featrelative,:],wall_touch[:,:],otherflies_vision[:,:]].T
    res['input'] = input[:-1,:]
    
  movement = Xfeat[:,1:]-Xfeat[:,:-1]
  movement[featorigin[0],:] = forward_vel
  movement[featorigin[1],:] = sideways_vel
  movement[featangle,...] = mabe.modrange(movement[featangle,...],-np.pi,np.pi)
  if simplify_out is not None:
    if simplify_out == 'global':
      movement = movement[featglobal,...]
    else:
      raise

  res['labels'] = movement.T
  res['init'] = Xfeat[featrelative==False,:2]
  res['scale'] = scale_perfly[:,id]
  res['nextinput'] = input[-1,:]
  
  # # check that we can undo correctly
  # res['labels'].T
  # thetavel = res['labels'][:,2]
  # theta0 = res['init'][2,0]
  # thetare = np.cumsum(np.r_[theta0[None],thetavel],axis=0)
  # np.max(np.abs(thetare-Xfeat[2,:]))
  # doriginrel = res['labels'][:,[1,0]]
  # dxy = mabe.rotate_2d_points(doriginrel,-thetare[:-1])
  # Xorigin[:,1:]-Xorigin[:,:-1]
  # np.max(np.sum(np.abs((Xorigin[:,1:]-Xorigin[:,:-1]).T-dxy),axis=0))
  
  if not smush:
    res['global'] = Xfeat[featrelative==False,:-1]
    res['relative'] = Xfeat[featrelative,:-1]
    if simplify_in == 'no_sensory':
      pass
    else:
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
  mabe.plot_flies(X[:,:,t,:],ax=ax,textlabels='fly',colors=np.zeros((X.shape[-1],3)))
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

def apply_mask(x,mask,nin=0):
  # mask with zeros
  x[mask,:-nin] = 0.
  x = torch.cat((x,mask[...,None].type(torch.float32)),dim=-1) 
  return x

def unzscore(x,mu,sig):
  return x*sig + mu
def zscore(x,mu,sig):
  return (x-mu)/sig

class FlyMLMDataset(torch.utils.data.Dataset):
  def __init__(self,data,max_mask_length=None,pmask=None,masktype='block',
               simplify_out=None,simplify_in=None):
    self.data = data
    self.max_mask_length = max_mask_length
    self.pmask = pmask
    self.masktype = masktype
    self.simplify_out = simplify_out # modulation of task to make it easier
    self.simplify_in = simplify_in
    
    self.mu_input = None
    self.sig_input = None
    self.mu_labels = None
    self.sig_labels = None
    
  def ismasked(self):
    return self.masktype is not None
    
  def zscore(self,mu_input=None,sig_input=None,mu_labels=None,sig_labels=None):
    
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
              
      return mu,sig

    if mu_input is None:
      self.mu_input,self.sig_input = zscore_helper(self.data,'input')
    else:
      self.mu_input = mu_input.copy()
      self.sig_input = sig_input.copy()
      
    if mu_labels is None:
      self.mu_labels,self.sig_labels = zscore_helper(self.data,'labels')
    else:
      self.mu_labels = mu_labels.copy()
      self.sig_labels = sig_labels.copy()      
    
    for example in self.data:
      example['input'] = ((example['input']-self.mu_input)/self.sig_input).astype(np.float32)
      example['labels'] = ((example['labels']-self.mu_labels)/self.sig_labels).astype(np.float32)
      
  def maskblock(self,inl):
    # choose a mask length
    maxl = min(inl-1,self.max_mask_length)
    l = np.random.randint(1,self.max_mask_length)
    
    # choose mask start
    t0 = np.random.randint(0,inl-l)
    t1 = t0+l
    
    # create mask
    mask = torch.zeros(inl,dtype=bool)
    mask[t0:t1] = True
    
    return mask
  
  def masklast(self,inl):
    mask = torch.zeros(inl,dtype=bool)
    mask[-1] = True
    return mask
  
  def maskind(self,inl):
    mask = torch.rand(inl)<=self.pmask
    if not torch.any(mask):
      imask = np.random.randint(inl)
      mask[imask] = True
    return mask
  
  def set_masktype(self,masktype):
    self.masktype = masktype
    
  def process_raw(self,input,labels,masktype='default'):
    ndim = input.ndim
    if ndim < 2:
      input = input[None,:]
    if labels.ndim < 2:
      labels = labels[None,:]
      
    if masktype == 'default':
      masktype = self.masktype
    
    if self.mu_input is not None:
      input = (input-self.mu_input)/self.sig_input
    input = torch.as_tensor(input.astype(np.float32).copy())
    if self.mu_labels is not None:
      labels = (labels-self.mu_labels)/self.sig_labels
    labels = torch.as_tensor(labels.astype(np.float32).copy())

    nin = input.shape[-1]
    contextl = input.shape[0]
    input = torch.cat((labels,input),dim=-1)
    if masktype == 'block':
      mask = self.maskblock(contextl)
    elif masktype == 'ind':
      mask = self.maskind(contextl)
    elif masktype == 'last':
      mask = self.masklast(contextl)
    else:
      mask = None
    if masktype is not None:
      input = apply_mask(input,mask,nin)
    return {'input': input, 'labels': labels, 'mask': mask}
    
  def input2pose(self,input=None):
    nlabels = self.data[0]['labels'].shape[-1]
    res = split_features(input[...,nlabels:],simplify=self.simplify_in)
    return res['pose']
    
  def __getitem__(self,idx):
    input = torch.as_tensor(self.data[idx]['input'])
    labels = torch.as_tensor(self.data[idx]['labels'].copy())
    if self.ismasked():
      nin = input.shape[-1]
      contextl = input.shape[0]
      input = torch.cat((labels,input),dim=-1)
      init = torch.as_tensor(self.data[idx]['init'][:,0].copy())
    else:
      # input: motion to frame t, pose, sensory frame t
      input = torch.cat((labels[:-1,:],input[1:,:]),dim=-1)
      # output: motion from t to t+1
      labels = labels[1:,:]
      init = torch.as_tensor(self.data[idx]['init'][:,1].copy())
      
    scale = torch.as_tensor(self.data[idx]['scale'].copy())
    categories = torch.as_tensor(self.data[idx]['categories'].copy())
    if self.masktype == 'block':
      mask = self.maskblock(contextl)
    elif self.masktype == 'ind':
      mask = self.maskind(contextl)
    elif self.masktype == 'last':
      mask = self.masklast(contextl)
    else:
      mask = None
    if self.masktype is not None:
      input = apply_mask(input,mask,nin)
    res = {'input': input, 'labels': labels, 
            'init': init, 'scale': scale, 'categories': categories,
            'metadata': self.data[idx]['metadata'].copy()}
    if self.masktype is not None:
      res['mask'] = mask
    return res
  
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
  
  def pred2pose(self,posein,globalin,pred):
    if self.mu_labels is not None:
      movement = unzscore(pred,self.mu_labels,self.sig_labels)
    else:
      movement = pred.copy()
    if self.simplify_out == 'global':
      posenext = posein
      globalnext = globalin+movement
    else:
      posenext = posein+movement[featrelative]
      movementglobal = movement[featglobal]
      originvelrel = movementglobal[[1,0],...]
      theta0 = globalin[2]
      originvel = mabe.rotate_2d_points(originvelrel[None,:],-theta0)
      globalnext = globalin.copy()
      globalnext[:2,...] += originvel.flatten()
      globalnext[2,...] += movementglobal[2,...]
      #globalnext = globalin+movement[featglobal]
    return posenext,globalnext
  
  def get_Xfeat(self,feat0,global0,movements):
    nlabel = movements.shape[-1]
    if self.masktype is not None:
      feat0 = feat0[...,:-1]
    feat0 = feat0[...,nlabel:]
    if self.mu_input is not None:
      feat0 = unzscore(feat0,self.mu_input,self.sig_input)
      movements = unzscore(movements,self.mu_labels,self.sig_labels)
      
    feat0 = split_features(feat0,simplify=self.simplify_in)
    Xorigin0 = global0[:2,...]
    Xtheta0 = global0[2,...] 

    thetavel = movements[...,feattheta]
    
    Xtheta = np.cumsum(np.r_[Xtheta0[None,...],thetavel],axis=0)
    Xoriginvelrel = movements[...,[1,0]]
    Xoriginvel = mabe.rotate_2d_points(Xoriginvelrel,-Xtheta[:-1,...])
    Xorigin = np.cumsum(np.r_[Xorigin0[None,...],Xoriginvel],axis=0)
    Xfeat = np.zeros((movements.shape[0]+1,nfeatures))
    Xfeat[:,featorigin] = Xorigin
    Xfeat[:,feattheta] = Xtheta

    if self.simplify_out == 'global':
      Xfeat[:,featrelative] = np.tile(feat0['pose'],(movements.shape[0]+1,1))
    else:
      Xfeatpose = np.cumsum(np.r_[feat0['pose'][None,:],movements[:,featrelative]],axis=0)
      Xfeat[:,featrelative] = Xfeatpose
    
    return Xfeat
  
  def get_Xkp(self,feat0,global0,movements,scale):
    Xfeat = self.get_Xfeat(feat0,global0,movements)
    Xkp = mabe.feat2kp(Xfeat.T[...,None],scale[...,None])
    return Xkp
  
  def get_global(self,global0,movements):
    if self.mu_input is not None:
      movements = unzscore(movements,self.mu_labels,self.sig_labels)
    if self.simplify_out is None or self.simplify_out == 'global':
      idxcenter = featorigin
      idxtheta = feattheta
    else:
      raise
    Xcenter = np.r_[global0[None,:2],movements[...,idxcenter]]
    Xcenter = np.cumsum(Xcenter,axis=0)
    Xtheta = np.r_[global0[None,2],movements[...,idxtheta]]
    Xtheta = mabe.modrange(np.cumsum(Xtheta,axis=0),-np.pi,np.pi)
    return Xcenter,Xtheta
  
  def get_outnames(self):
    outnames_global = ['forward','sideways','orientation']

    if self.simplify_out == 'global':
      outnames = outnames_global
    else:
      outnames = outnames_global + [mabe.posenames[x] for x in np.nonzero(featrelative)[0]]
    return outnames

def causal_criterion(tgt,pred):
  d = tgt.shape[-1]
  err = torch.sum(torch.abs(tgt-pred))/d
  return err
  
def masked_criterion(tgt,pred,mask):
  d = tgt.shape[-1]
  err = torch.sum(torch.abs(tgt[mask,:]-pred[mask,:]))/d
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

def save_model(savefile,model,lr_optimizer=None,scheduler=None,loss=None,val_loss=None):
  tosave = {'model':model.state_dict()}
  if lr_optimizer is not None:
    tosave['lr_optimizer'] = lr_optimizer.state_dict()
  if scheduler is not None:
    tosave['scheduler'] = scheduler.state_dict()
  if loss is not None:
    tosave['loss'] = loss
  if val_loss is not None:
    tosave['val_loss'] = val_loss
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
  val_loss = None
  if 'loss' in state:
    loss = state['loss']
  if 'val_loss' in state:
    val_loss = state['val_loss']
  return loss,val_loss

def debug_plot_batch_traj(example,train_dataset,pred=None,data=None,nsamplesplot=3,
                          h=None,ax=None,fig=None,label_true='True',label_pred='Pred'):
  batch_size = example['input'].shape[0]
  contextl = example['input'].shape[1]
    
  true_color = [0,0,0]
  true_color_y = [.5,.5,.5]
  pred_cmap = lambda x: plt.get_cmap("tab10")(x%10)
  
  if 'mask' in example:
    mask = example['mask']
  else:
    mask = None
  
  if ax is None:
    fig,ax = plt.subplots(1,nsamplesplot)
    
  samplesplot = np.round(np.linspace(0,batch_size-1,nsamplesplot)).astype(int)
  for i in range(nsamplesplot):
    iplot = samplesplot[i]
    Xcenter_true,Xtheta_true = train_dataset.get_global(example['init'][iplot,...].numpy(),
                                                        example['labels'][iplot,...].numpy())
    Xcenter_true0 = Xcenter_true[[0,],:]
    Xcenter_true = Xcenter_true - Xcenter_true0
    zmovement_true = example['labels'][iplot,...].numpy()
    err_movement = None
    total_err_movement = None
    if mask is not None:
      maskcurr = np.zeros(mask.shape[-1]+1,dtype=bool)
      maskcurr[:-1] = mask[iplot,...].numpy()
    else:
      maskcurr = np.zeros(example['labels'].shape[-2]+1,dtype=bool)
      maskcurr[:-1] = True
    maskidx = np.nonzero(maskcurr)[0]
    if pred is not None:
      Xcenter_pred,Xtheta_pred = train_dataset.get_global(example['init'][iplot,...].numpy(),
                                                          pred[iplot,...].numpy())
      Xcenter_pred = Xcenter_pred - Xcenter_true0

      zmovement_pred = pred[iplot,...].numpy()
      d = example['labels'].shape[-1]
      if mask is not None:
        nmask = np.count_nonzero(maskcurr)
      else:
        nmask = np.prod(example['labels'].shape[:-1])
      err_movement = torch.abs(example['labels'][iplot,maskidx,:]-pred[iplot,maskidx,:])/nmask
      total_err_movement = torch.sum(err_movement).item()/d

    elif data is not None:
      t0 = example['metadata']['t0'][iplot].item()
      flynum = example['metadata']['flynum'][iplot].item()
      
      Xcenter_pred,Xtheta_pred = compute_global(data['X'][...,t0:t0+contextl+1,flynum])
      Xcenter_pred = Xcenter_pred[...,0].T
      Xcenter_pred = Xcenter_pred - Xcenter_true0
      Xtheta_pred = Xtheta_pred[...,0]
      zmovement_pred = None
    else:
      Xcenter_pred = None
      Xtheta_pred = None
      zmovement_pred = None
      
    #ax[i,0].cla()
    #ax[i,0].plot(Xcenter_true[:,0]-1,'-',color=true_color,label=label_true+' x')
    #ax[i,0].plot(Xcenter_true[:,1]+1,'-',color=true_color_y,label=label_true+' y')
    #if mask is not None:
    #  ax[i,0].plot(maskidx,Xcenter_true[maskcurr,0]-1,'o',color=true_color)
    #  ax[i,0].plot(maskidx,Xcenter_true[maskcurr,1]+1,'o',color=true_color_y)
    #if Xcenter_pred is not None:
    #  ax[i,0].plot(Xcenter_pred[:,0]-1,'-',color=pred_cmap(0),label=label_pred+' x')
    #  ax[i,0].plot(Xcenter_pred[:,1]+1,'-',color=pred_cmap(1),label=label_pred+' y')
    #  if mask is not None:
    #    ax[i,0].plot(maskidx,Xcenter_pred[maskcurr,0]-1,'o',color=pred_cmap(0))
    #    ax[i,0].plot(maskidx,Xcenter_pred[maskcurr,1]+1,'o',color=pred_cmap(1))
    #ax[i,0].legend()
    #ax[i,1].cla()
    #ax[i,1].plot(Xtheta_true,'-',color=true_color,label=label_true,lw=2)
    #if mask is not None:
    #  ax[i,1].plot(maskidx,Xtheta_true[maskcurr],'o',color=true_color)
    #ax[i,1].set_xlabel('Frame')
    #ax[i,1].set_ylabel('Center (mm)')
      
    #if Xtheta_pred is not None:
    #  ax[i,1].plot(Xtheta_pred,'-',color=pred_cmap(0),label=label_pred,lw=1)
    #  if mask is not None:
    #    ax[i,1].plot(maskidx,Xtheta_pred[maskcurr],'o',color=pred_cmap(0))
    #ax[i,1].legend()
    #ax[i,1].set_xlabel('Frame')
    #ax[i,1].set_ylabel('Orientation (rad)')
    #ax[i,1].legend()
    mult = 2.
    nout = zmovement_true.shape[1]
    outnames = train_dataset.get_outnames()

    ax[i].cla()
    for feati in range(nout):
      ax[i].plot([0,contextl],[mult*feati,]*2,':',color=[.5,.5,.5])
      ax[i].plot(mult*feati + zmovement_true[:,feati],'-',color=true_color,label=f'{outnames[feati]}, true')
      ax[i].plot(maskidx,mult*feati + zmovement_true[maskcurr[:-1],feati],'o',color=true_color,label=f'{outnames[feati]}, true')

      labelcurr = outnames[feati]
      if zmovement_pred is not None:
        h = ax[i].plot(mult*feati + zmovement_pred[:,feati],'--',label=f'{outnames[feati]}, pred',color=pred_cmap(feati))
        ax[i].plot(maskidx,mult*feati + zmovement_pred[maskcurr[:-1],feati],'o',color=pred_cmap(feati),label=f'{outnames[feati]}, pred')
        labelcurr = f'{outnames[feati]} {err_movement[0,feati].item(): .2f}'
      ax[i].text(0,mult*(feati+.5),labelcurr,horizontalalignment='left',verticalalignment='top')

    if zmovement_pred is not None:
      ax[i].set_title(f'Err = {total_err_movement: .2f}')
    ax[i].set_xlabel('Frame')
    ax[i].set_ylabel('Z-scored movement')
    ax[i].set_ylim([-mult,mult*(nout)])

  return ax,fig

def debug_plot_batch_pose(example,train_dataset,pred=None,data=None,ntsplot=5,nsamplesplot=3,h=None,ax=None,fig=None):
 
  batch_size = example['input'].shape[0]
  contextl = example['input'].shape[1]
  
  if ax is None:
    fig,ax = plt.subplots(nsamplesplot,ntsplot)
    
  tsplot = np.round(np.linspace(0,contextl,ntsplot)).astype(int)
  samplesplot = np.round(np.linspace(0,batch_size-1,nsamplesplot)).astype(int)
  if h is None:
    h = {'kpt0': [], 'kpt1': [], 'edge0': [], 'edge1': []}
  for i in range(nsamplesplot):
    iplot = samplesplot[i]
    Xkp_true = train_dataset.get_Xkp(example['input'][iplot,0,...].numpy(),
                                example['init'][iplot,...].numpy(),
                                example['labels'][iplot,...].numpy(),
                                example['scale'][iplot,...].numpy())
    Xkp_true = Xkp_true[...,0]
    if pred is not None:
      Xkp_pred = train_dataset.get_Xkp(example['input'][iplot,0,...].numpy(),
                                  example['init'][iplot,...].numpy(),
                                  pred[iplot,...].numpy(),
                                  example['scale'][iplot,...].numpy())
      Xkp_pred = Xkp_pred[...,0]
    elif data is not None:
      t0 = example['metadata']['t0'][iplot].item()
      flynum = example['metadata']['flynum'][iplot].item()
      Xkp_pred = data['X'][:,:,t0:t0+contextl+1,flynum]
    else:
      Xkp_pred = None
    for key in h.keys():
      if len(h[key]) <= i:
        h[key].append([None,]*ntsplot)
    for j in range(ntsplot):
      tplot = tsplot[j]
      ax[i,j].set_title(f't = {tplot}, sample = {iplot}')
      h['kpt0'][i][j],h['edge0'][i][j],_,_,_ = mabe.plot_fly(Xkp_true[:,:,tplot],
                                                             skel_lw=2,color=[0,0,0],
                                                             ax=ax[i,j],hkpt=h['kpt0'][i][j],hedge=h['edge0'][i][j])
      if Xkp_pred is not None:
        h['kpt1'][i][j],h['edge1'][i][j],_,_,_ = mabe.plot_fly(Xkp_pred[:,:,tplot],
                                                              skel_lw=1,color=[0,1,1],
                                                              ax=ax[i,j],hkpt=h['kpt1'][i][j],hedge=h['edge1'][i][j])      
      ax[i,j].set_aspect('equal')
      ax[i,j].relim()
      ax[i,j].autoscale()

  return h,ax,fig
  
# location of data
datadir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022'
intrainfile = os.path.join(datadir,'usertrain.npz')
invalfile = os.path.join(datadir,'testtrain.npz')
categories = ['71G01','male',]
#inchunkfile = os.path.join(datadir,'chunk_usertrain.npz')
inchunkfile = None
#savechunkfile = os.path.join(datadir,f'chunk_usertrain_{"_".join(categories)}.npz')
savechunkfile = None
savedir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets'
# MLM - no sensory
#loadmodelfile = os.path.join(savedir,'flymlm_71G01_male_epoch100_202301215712.pth')
# MLM with sensory
#loadmodelfile = os.path.join(savedir,'flymlm_71G01_male_epoch100_202301003317.pth')
# CLM with sensory
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_202301211242.pth')
# CLM with sensory but only global motion output
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch15_202301014322.pth')
#loadmodelfile = None
# CLM, predicting forward, sideways vel
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch60_202302231241.pth')
loadmodelfile = None

# parameters

narena = 2**10
theta_arena = np.linspace(-np.pi,np.pi,narena+1)[:-1]

# sequence length per training example
#CONTEXTL = 512
CONTEXTL = 65
# type of model to train
#modeltype = 'mlm'
MODELTYPE = 'clm'
# masking method
MASKTYPE = 'ind'
if MODELTYPE == 'clm':
  MASKTYPE = None
# how long to mask out at a time (maximum), masktype = 'block'
MAX_MASK_LENGTH = 4
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
MODEL_ARGS = {'d_model': 2048, 'nhead': 8, 'd_hid': 512, 'nlayers': 6, 'dropout': .3}
# whether to try to simplify the task, and how
SIMPLIFY_OUT = None #'global'
SIMPLIFY_IN = None
#SIMPLIFY_IN = 'no_sensory'

NPLOT = 32*512*5 // (BATCH_SIZE*CONTEXTL)
SAVE_EPOCH = 5

NUMPY_SEED = 123
TORCH_SEED = 456

np.random.seed(NUMPY_SEED)
torch.manual_seed(TORCH_SEED)
device = torch.device('cuda')

plt.ion()

# load in data
data = load_raw_npz_data(intrainfile)
valdata = load_raw_npz_data(invalfile)

# filter out data
if categories is not None and len(categories) > 0:
  filter_data_by_categories(data,categories)
  filter_data_by_categories(valdata,categories)

# compute scale parameters
scale_perfly = compute_scale_allflies(data)
val_scale_perfly = compute_scale_allflies(valdata)

# throw out data that is missing scale information - not so many frames
idsremove = np.nonzero(np.any(np.isnan(scale_perfly),axis=0))[0]
data['isdata'][np.isin(data['ids'],idsremove)] = False
idsremove = np.nonzero(np.any(np.isnan(val_scale_perfly),axis=0))[0]
valdata['isdata'][np.isin(valdata['ids'],idsremove)] = False

# function for computing features
reparamfun = lambda x,id,flynum: compute_features(x,id,flynum,scale_perfly,SENSORY_PARAMS,outtype=np.float32,
                                                  simplify_out=SIMPLIFY_OUT,simplify_in=SIMPLIFY_IN)

val_reparamfun = lambda x,id,flynum,**kwargs: compute_features(x,id,flynum,val_scale_perfly,
                                                               SENSORY_PARAMS,outtype=np.float32,
                                                               simplify_out=SIMPLIFY_OUT,simplify_in=SIMPLIFY_IN,**kwargs)
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


print('Chunking val data...')
valX = chunk_data(valdata,CONTEXTL,val_reparamfun)
print('Done.')

train_dataset = FlyMLMDataset(X,max_mask_length=MAX_MASK_LENGTH,pmask=PMASK,masktype=MASKTYPE,simplify_out=SIMPLIFY_OUT)
train_dataset.zscore()
val_dataset = FlyMLMDataset(valX,max_mask_length=MAX_MASK_LENGTH,pmask=PMASK,
                            masktype=MASKTYPE,simplify_out=SIMPLIFY_OUT)
val_dataset.zscore(train_dataset.mu_input,train_dataset.sig_input,
                   train_dataset.mu_labels,train_dataset.sig_labels)

d_feat = train_dataset[0]['input'].shape[-1]
d_output = train_dataset[0]['labels'].shape[-1]

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               pin_memory=True,
                                              )
ntrain = len(train_dataloader)

val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             pin_memory=True,
                                             )
nval = len(val_dataloader)

# spot check that we can get poses from examples
example = next(iter(train_dataloader))

fig,ax = plt.subplots(3,2)
inputidxstart = [0,]
inputidxtype = ['movement']
inputidxstart.append(inputidxstart[-1]+example['labels'].shape[-1])
inputidxtype.append('pose')
if SIMPLIFY_IN != 'no_sensory':
  inputidxstart.append(inputidxstart[-1]+np.count_nonzero(featrelative))
  inputidxtype.append('otherflies_vision')
  inputidxstart.append(inputidxstart[-1]+SENSORY_PARAMS['n_oma'])
  inputidxtype.append('wall_touch')
  
for iplot in range(3):
  ax[iplot,0].cla()
  ax[iplot,1].cla()
  ax[iplot,0].imshow(example['input'][iplot,...],vmin=-3,vmax=3,cmap='coolwarm')
  ax[iplot,0].axis('auto')
  ax[iplot,0].set_title(f'Input {iplot}')
  #ax[iplot,0].set_xticks(inputidxstart)
  for j in range(len(inputidxtype)):
    ax[iplot,0].plot([inputidxstart[j],]*2,[-.5,example['input'].shape[1]-.5],'k-')
    ax[iplot,0].text(inputidxstart[j],CONTEXTL-1,inputidxtype[j],horizontalalignment='left')
  #ax[iplot,0].set_xticklabels(inputidxtype)
  ax[iplot,1].imshow(example['labels'][iplot,...],vmin=-3,vmax=3,cmap='coolwarm')
  ax[iplot,1].axis('auto')
  ax[iplot,1].set_title(f'Labels {iplot}')
  

h,ax,fig = debug_plot_batch_pose(example,train_dataset,data=data)
ax[-1,0].set_xlabel('Train')
axtraj,figtraj = debug_plot_batch_traj(example,train_dataset,data=data,
                                       label_true='Label',
                                       label_pred='Raw')
axtraj[0].set_title('Train')
example = next(iter(val_dataloader))
valh,valax,valfig = debug_plot_batch_pose(example,val_dataset,data=valdata)
valax[-1,0].set_xlabel('Val')
valaxtraj,valfigtraj = debug_plot_batch_traj(example,val_dataset,data=valdata,
                                       label_true='Label',
                                       label_pred='Raw')
valaxtraj[0].set_title('Val')
plt.show()
plt.pause(.001)

model = TransformerModel(d_feat,d_output,**MODEL_ARGS).to(device)

num_training_steps = NUM_TRAIN_EPOCHS * ntrain
optimizer = transformers.optimization.AdamW(model.parameters(),**OPTIMIZER_ARGS)
lr_scheduler = transformers.get_scheduler('linear',optimizer,num_warmup_steps=0,
                                          num_training_steps=num_training_steps)

train_loss_epoch = torch.zeros(NUM_TRAIN_EPOCHS)
train_loss_iter = torch.zeros(num_training_steps)
val_loss_epoch = torch.zeros(NUM_TRAIN_EPOCHS)
train_loss_epoch[:] = torch.nan
train_loss_iter[:] = torch.nan
val_loss_epoch[:] = torch.nan
last_val_loss = None

progress_bar = tqdm.tqdm(range(num_training_steps))

contextl = example['input'].shape[1]
if MODELTYPE == 'mlm':
  train_src_mask = generate_square_full_mask(contextl).to(device)
elif MODELTYPE == 'clm':
  train_src_mask = generate_square_subsequent_mask(contextl).to(device)
else:
  raise

if loadmodelfile is None:
  
  lossfig = plt.figure()
  lossax = plt.gca()
  htrainloss, = lossax.plot(train_loss_epoch,'.-',label='Train')
  hvalloss, = lossax.plot(val_loss_epoch,'.-',label='Val')
  lossax.set_xlabel('Epoch')
  lossax.set_ylabel('Loss')
  
  savetime = datetime.datetime.now()
  savetime = savetime.strftime('%Y%m%H%M%S')

  for epoch in range(NUM_TRAIN_EPOCHS):
    
    model.train()
    tr_loss = torch.tensor(0.0).to(device)

    nmask_train = 0
    for step, example in enumerate(train_dataloader):
      
      pred = model(example['input'].to(device=device),train_src_mask)
      if MODELTYPE == 'mlm':
        loss = masked_criterion(example['labels'].to(device=device),pred,
                                example['mask'].to(device))
      elif MODELTYPE == 'clm':
        loss = causal_criterion(example['labels'].to(device=device),pred)
        
      loss.backward()
      if MODELTYPE == 'mlm':
        nmask_train += torch.count_nonzero(example['mask'])
      else:
        nmask_train += BATCH_SIZE*contextl

      if step % NPLOT == 0:
        debug_plot_batch_pose(example,train_dataset,pred=pred.detach().cpu(),h=h,ax=ax,fig=fig)
        debug_plot_batch_traj(example,train_dataset,pred=pred.detach().cpu(),ax=axtraj,fig=figtraj)
        axtraj[0].set_title('Train')
        valexample = next(iter(val_dataloader))
        with torch.no_grad():
          valpred = model(valexample['input'].to(device=device),train_src_mask)
          debug_plot_batch_pose(valexample,val_dataset,pred=valpred.cpu(),h=valh,ax=valax,fig=valfig)
          debug_plot_batch_traj(valexample,val_dataset,pred=valpred.cpu(),ax=valaxtraj,fig=valfigtraj)
          valaxtraj[0].set_title('Val')

        plt.pause(.01)

      tr_loss_step = loss.detach()
      tr_loss += tr_loss_step

      # gradient clipping
      torch.nn.utils.clip_grad_norm_(model.parameters(),MAX_GRAD_NORM)
      optimizer.step()
      lr_scheduler.step()
      model.zero_grad()
      progress_bar.set_postfix({'train loss': tr_loss.item()/nmask_train,'last val loss': last_val_loss,'epoch': epoch})
      progress_bar.update(1)

    train_loss_epoch[epoch] = tr_loss.item() / nmask_train

    model.eval()
    with torch.no_grad():
      val_loss = torch.tensor(0.0).to(device)
      nmask_val = 0
      for example in val_dataloader:
        pred = model(example['input'].to(device=device),train_src_mask)
        
        if MODELTYPE == 'mlm':
          loss = masked_criterion(example['labels'].to(device=device),pred,
                                  example['mask'].to(device))
          nmask_val += torch.count_nonzero(example['mask'])
        elif MODELTYPE == 'clm':
          loss = causal_criterion(example['labels'].to(device=device),pred)
          nmask_val += BATCH_SIZE*contextl

        val_loss+=loss
        
      val_loss_epoch[epoch] = val_loss.item() / nmask_val

      last_val_loss = val_loss_epoch[epoch]

    htrainloss.set_data(np.arange(epoch+1),train_loss_epoch[:epoch+1])
    hvalloss.set_data(np.arange(epoch+1),val_loss_epoch[:epoch+1])
    lossax.relim()
    lossax.autoscale()
    plt.pause(.01)

    # rechunk the training data
    print(f'Rechunking data after epoch {epoch}')
    X = chunk_data(data,CONTEXTL,reparamfun)

    train_dataset = FlyMLMDataset(X,max_mask_length=MAX_MASK_LENGTH,pmask=PMASK,masktype=MASKTYPE,simplify_out=SIMPLIFY_OUT)
    train_dataset.zscore()
    print('New training data set created')

    if (epoch+1)%SAVE_EPOCH == 0:
      savefile = os.path.join(savedir,f'fly{MODELTYPE}_{"_".join(categories)}_epoch{epoch+1}_{savetime}.pth')
      print(f'Saving to file {savefile}')
      save_model(savefile,model,lr_optimizer=optimizer,scheduler=lr_scheduler,loss=train_loss_epoch,val_loss=val_loss_epoch)


  savefile = os.path.join(savedir,f'fly{MODELTYPE}_{"_".join(categories)}_epoch{epoch+1}_{savetime}.pth')
  save_model(savefile,model,lr_optimizer=optimizer,scheduler=lr_scheduler,loss=train_loss_epoch,val_loss=val_loss_epoch)

  print('Done training')
else:
  train_loss_epoch,val_loss_epoch = load_model(loadmodelfile,model,lr_optimizer=optimizer,scheduler=lr_scheduler)

lossfig = plt.figure()
lossax = plt.gca()
htrainloss = lossax.plot(train_loss_epoch.cpu(),label='Train')
hvalloss = lossax.plot(val_loss_epoch.cpu(),label='Val')
lossax.set_xlabel('Epoch')
lossax.set_ylabel('Loss')
lossax.legend()

model.eval()

all_pred = []
all_mask = []
all_labels = []
with torch.no_grad():
  val_loss = torch.tensor(0.0).to(device)
  for example in val_dataloader:
    pred = model(example['input'].to(device=device),train_src_mask)
    all_pred.append(pred.cpu())
    if 'mask' in example:
      all_mask.append(example['mask'])
    all_labels.append(example['labels'])

nplot = min(len(all_labels),8000//BATCH_SIZE//CONTEXTL+1)

def stackhelper(all_pred,all_labels,all_mask,nplot):

  predv = torch.stack(all_pred[:nplot],dim=0)
  if len(all_mask) > 0:
    maskv = torch.stack(all_mask[:nplot],dim=0)
  else:
    maskv = None
  labelsv = torch.stack(all_labels[:nplot],dim=0)
  s = list(predv.shape)
  s[2] = 1
  nan = torch.zeros(s,dtype=predv.dtype)
  nan[:] = torch.nan
  predv = torch.cat((predv,nan),dim=2)
  predv = predv.reshape((predv.shape[0]*predv.shape[1]*predv.shape[2],predv.shape[3]))
  labelsv = torch.cat((labelsv,nan),dim=2)
  labelsv = labelsv.reshape((labelsv.shape[0]*labelsv.shape[1]*labelsv.shape[2],labelsv.shape[3]))
  if maskv is not None:
    maskv = torch.cat((maskv,torch.zeros(s[:-1],dtype=bool)),dim=2)
    maskv = maskv.flatten()
  
  return predv,labelsv,maskv  

predv,labelsv,maskv = stackhelper(all_pred,all_labels,all_mask,nplot)

if MODELTYPE == 'mlm':
  
  maskidx = torch.nonzero(maskv)[:,0]

  all_pred = []
  all_mask = []
  all_labels = []
  val_dataset.set_masktype('last')
  with torch.no_grad():
    val_loss = torch.tensor(0.0).to(device)
    nmask_val = 0
    for example in val_dataloader:
      pred = model(example['input'].to(device=device),train_src_mask)
      all_pred.append(pred.cpu())
      all_mask.append(example['mask'])
      all_labels.append(example['labels'])

  predv_last,labelsv_last,maskv_last = stackhelper(all_pred,all_labels,all_mask,nplot)
  
  maskidx_last = torch.nonzero(maskv_last)[:,0]

if MODELTYPE == 'mlm':
  nax = 2
else:
  nax = 1
fig,ax = plt.subplots(d_output,nax,sharex='all',sharey='row',squeeze=False)
fig.set_figheight(20)
fig.set_figwidth(20)
toff = 0
pred_cmap = lambda x: plt.get_cmap("tab10")(x%10)
outnames = val_dataset.get_outnames()
for i in range(d_output):
  ax[i,0].cla()
  ax[i,0].plot(labelsv[:,i],'k-',label='True')
  if MODELTYPE == 'mlm':
    ax[i,0].plot(maskidx,predv[maskv,i],'.',color=pred_cmap(i),label='Pred')
  else:
    ax[i,0].plot(predv[:,i],'-',color=pred_cmap(i),label='Pred')
  ax[i,0].set_ylim([-3,3])
  if nax > 1:
    ax[i,1].cla()
    ax[i,1].plot(labelsv_last[:,i],'k-',label='True')
    ax[i,1].plot(maskidx_last,predv_last[maskv_last,i],'.',color=pred_cmap(i),label='Pred')
  ax[i,0].set_xlim([0,labelsv.shape[0]])
  ti = ax[i,0].set_title(outnames[i],y=1.0,pad=-14,color=pred_cmap(i),loc='left')
  plt.setp(ti,color=pred_cmap(i))
  
if nax > 1:
  ax[0,0].set_title('Random masking')
  ax[0,1].set_title('Last masking')
plt.tight_layout(h_pad=0)

DEBUG = False
contextl = CONTEXTL
contextlpad = contextl + 1
if MODELTYPE == 'mlm':
  masktype = 'last'
  mask = val_dataset.masklast(contextl)
else:
  masktype = None
  mask = None

model.eval()

# all frames for the main fly must have real data
allisdata = interval_all(valdata['isdata'],contextlpad)
isnotsplit = interval_all(valdata['isstart']==False,contextlpad-1)[1:,...]
canstart = np.logical_and(allisdata,isnotsplit)
flynum = 0
TPRED = 1000
t0 = np.nonzero(canstart[:,flynum])[0][360-CONTEXTL]
id = valdata['ids'][t0,flynum]
scale = val_scale_perfly[:,id]

# where to store predictions
# real data for other flies
# nan after initial contextl frames
Xkp_pred = valdata['X'][...,t0:t0+TPRED,:].copy()

# store groundtruth
Xfeat_true = compute_features(Xkp_pred,id,flynum,val_scale_perfly,SENSORY_PARAMS,
                              outtype=np.float32,simplify_out=SIMPLIFY_OUT,
                              simplify_in=SIMPLIFY_IN,smush=False)

Xkp_pred[:,:,contextl+1:,flynum] = np.nan

# frame we will predict motion from
trel = contextl - 1
t = t0 + trel

# contextl frames
# frames trel = 0, 1, ..., contextl-1
Xkpin = Xkp_pred[:,:,trel-contextl+1:trel+1,:]
# contextl-1 frames
# input corresponds to frames trel = 0, 1, ..., contextl-2
# nextinput corresponds to frame contextl-1
# labels corresponds to motions:
# frame 0 to 1
# frame 1 to 2
# ...
# frame contextl-3 to contextl-2
# global corresponds to frames trel = 0, 1, ..., contextl-2
Xfeatin = compute_features(Xkpin,id,flynum,val_scale_perfly,SENSORY_PARAMS,
                           outtype=np.float32,simplify_out=SIMPLIFY_OUT,
                           simplify_in=SIMPLIFY_IN,smush=False)
Xfeatin.pop('init')
if MODELTYPE == 'clm':
  # inputs correspond to frames trel = 1,...,contextl-2
  Xfeatin['input'] = Xfeatin['input'][1:,:]
  for key in ['global','relative','wall_touch','otherflies_vision']:
    Xfeatin[key] = Xfeatin[key][:,1:]
  

for trel in tqdm.trange(contextl-1,TPRED-1):
  t = t0 + trel

  # motion from frame contextl-1 to contextl
  movementlast = Xfeatin['labels'][-1,:]
  # input at frame: contextl-1
  rawinputcurr = Xfeatin['nextinput']  
  
  # global position at frame contextl-2
  global0 = Xfeatin['global'][:,-1]
  Xorigin0 = global0[:2]
  Xtheta0 = global0[2]
  Xoriginvelrel = movementlast[[1,0]]
  Xoriginvel = mabe.rotate_2d_points(Xoriginvelrel[None,:],-Xtheta0)
  Xorigincurr = Xorigin0 + Xoriginvel
  
  assert(np.any(np.isnan(rawinputcurr))==False)
  assert(np.any(np.isnan(movementlast))==False)
  
  # global position for frame contextl-1
  globalcurr = Xfeatin['global'][:,-1]+movementlast[featglobal]
  globalcurr[:2] = Xorigincurr
  
  # concatenate inputs
  # MLM: rawinput corresponds to frames 0 to contextl-1
  # CLM: rawinput corresponds to frames 1 to contextl-1
  rawinput = np.r_[Xfeatin['input'],rawinputcurr[None,:]]
  if MODELTYPE == 'mlm':
    # motion 0->1, 1->2, ..., contextl-2->contexl-1, dummy
    movementin = np.r_[Xfeatin['labels'],np.zeros((1,Xfeatin['labels'].shape[-1]))]
    movementin[-1,:] = np.nan
  else:
    # motion 0->1, 1->2, ..., contextl-2->contexl-1
    movementin = Xfeatin['labels']
  res = val_dataset.process_raw(rawinput,movementin,masktype=masktype)
  input = res['input']
  # predicting motion from frame contexl-1->contextl

  if DEBUG:
    # true motion from frame contextl-1->contextl
    movementtrue = Xfeat_true['labels'][trel,:].copy().astype(np.float32)
    pred = zscore(movementtrue,val_dataset.mu_labels,val_dataset.sig_labels)
    pred = torch.tensor(pred)
  else:
    with torch.no_grad():
      pred = model(input[None,...].to(device),train_src_mask)
      # movement from contextl-1 to contextl
      pred = pred[0,-1,:].cpu()

  # input at frame: contextl-1
  res = split_features(rawinputcurr,simplify=SIMPLIFY_IN)
  posecurr = res['pose']
  # predicted pose at frame contextl
  posenext,globalnext = val_dataset.pred2pose(posecurr,globalcurr,pred.numpy())
  featnext = np.zeros(nfeatures,dtype=posenext.dtype)
  featnext[featrelative] = posenext
  featnext[featglobal] = globalnext
  # kp for frame contextl
  Xkp_next = mabe.feat2kp(featnext,scale)
  Xkp_next = Xkp_next[:,:,0,0]
  Xkp_pred[:,:,trel+1,flynum] = Xkp_next

  # Xfeatnext:
  # input: frame contextl-1
  # labels: movement from contextl-1 to contextl
  Xfeatnext = compute_features(Xkp_pred[:,:,[trel,trel+1],:],id,flynum,val_scale_perfly,SENSORY_PARAMS,
                              outtype=np.float32,simplify_out=SIMPLIFY_OUT,
                              simplify_in=SIMPLIFY_IN,smush=False)

  # MLM: input corresponds to frame 1 to contextl-1
  # CLM: input corresponds to frame 2 to contextl-1
  # labels corresponds to 
  # motion from 1 to 2
  # ...
  # motion from frame contextl-1 to contextl
  Xfeatin['input'] = np.r_[Xfeatin['input'][1:,:],Xfeatnext['input']]
  Xfeatin['labels'] = np.r_[Xfeatin['labels'][1:,:],Xfeatnext['labels']]
  Xfeatin['global'] = np.c_[Xfeatin['global'][:,1:],Xfeatnext['global']]
  Xfeatin['relative'] = np.c_[Xfeatin['relative'][:,1:],Xfeatnext['relative']]
  Xfeatin['nextinput'] = Xfeatnext['nextinput']

isreal = mabe.get_real_flies(Xkp_pred)
Xkp_pred = Xkp_pred[...,isreal]

# indices of other flies
idxother = np.ones(Xkp_pred.shape[-1],dtype=bool)
idxother[flynum] = False
idxother = np.nonzero(idxother)[0]

h = {}

trel = 0
t = t0+trel
fig,ax = plt.subplots(1,2)
fig.set_figheight(11.4)
fig.set_figwidth(22.3)
h['kpt_other_pred'],h['edge_other_pred'],_,_,_ = mabe.plot_flies(Xkp_pred[...,trel,idxother],ax=ax[0])
ax[0].set_aspect('equal')
h['kpt_main_pred'],h['edge_main_pred'],_,_,_ = mabe.plot_fly(Xkp_pred[...,trel,flynum],ax=ax[0],
                                                             color='k',skel_lw=3,kpt_ms=12)
ax[0].set_title(f'Initialize, t = {t}')
mabe.plot_arena(ax=ax[0])
h['kpt_other_true'],h['edge_other_true'],_,_,_ = mabe.plot_flies(valdata['X'][...,t,idxother],ax=ax[1])
h['kpt_main_true'],h['edge_main_true'],_,_,_ = mabe.plot_fly(valdata['X'][...,t,flynum],
                                                             color='k',skel_lw=3,kpt_ms=12,ax=ax[1])
mabe.plot_arena(ax=ax[1])
ax[1].set_aspect('equal')
ax[1].set_title(f'True')

minv = -mabe.ARENA_RADIUS_MM*1.01
maxv = mabe.ARENA_RADIUS_MM*1.01
ax[0].set_xlim(minv,maxv)
ax[0].set_ylim(minv,maxv)
ax[1].set_xlim(minv,maxv)
ax[1].set_ylim(minv,maxv)
ax[1].set_title('True')
plt.tight_layout()

def update(trel):

  t = t0+trel

  mabe.plot_flies(Xkp_pred[...,trel+1,idxother],ax=ax[0],
                  hkpts=h['kpt_other_pred'],hedges=h['edge_other_pred'])

  mabe.plot_fly(Xkp_pred[...,trel+1,flynum],ax=ax[0],
                color='k',skel_lw=3,kpt_ms=12,
                hkpt=h['kpt_main_pred'],hedge=h['edge_main_pred'])
  if trel < contextl:
    h['ti_pred'] = ax[0].set_title(f'Initialize, t = {t}')
  else:
    h['ti_pred'] = ax[0].set_title(f'Pred, t = {t}')
  mabe.plot_flies(valdata['X'][...,t+1,idxother],ax=ax[1],
                  hkpts=h['kpt_other_true'],hedges=h['edge_other_true'])
  mabe.plot_fly(valdata['X'][...,t+1,flynum],
                color='k',skel_lw=3,kpt_ms=12,ax=ax[1],
                hkpt=h['kpt_main_true'],hedge=h['edge_main_true'])
  
  hlist = []
  for hcurr in h.values():
    if type(hcurr) == list:
      hlist+=hcurr
    else:
      hlist+=[hcurr,]
  return hlist

tstart = max(0,CONTEXTL-10)
ani = animation.FuncAnimation(fig, update, frames=np.arange(tstart,TPRED-1,1,dtype=int))

vidtime = datetime.datetime.now().strftime('%Y%m%H%M%S')

savevidfile = os.path.join(savedir,f'samplevideo_{"_".join(categories)}_{vidtime}.gif')
  
print('Saving animation to file %s...'%savevidfile)
writer = animation.PillowWriter(fps=30)
ani.save(savevidfile,writer=writer)
print('Finished writing.')
