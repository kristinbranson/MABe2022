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
# import sklearn.preprocessing
import sklearn.cluster

legtipnames = [
  'right_front_leg_tip',
  'right_middle_leg_tip',
  'right_back_leg_tip',
  'left_back_leg_tip',
  'left_middle_leg_tip',
  'left_front_leg_tip',
]

SENSORY_PARAMS = {
  'n_oma': 72,
  'inner_arena_radius': 17.5, # in mm
  'outer_arena_radius': mabe.ARENA_RADIUS_MM,
  'arena_height': 3.5,
  'otherflies_vision_exp': .6,
  'wallkpnames': mabe.keypointnames,
  #'wallkpnames': legtipnames,
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
  
def debug_plot_wall_touch(t,xwall,ywall,distleg,wall_touch):
  plt.figure()
  plt.clf()
  ax = plt.subplot(1,2,1)
  ax.plot(xwall.flatten(),ywall.flatten(),'k.')
  theta_arena = np.linspace(-np.pi,np.pi,100)
  ax.plot(np.cos(theta_arena)*SENSORY_PARAMS['inner_arena_radius'],np.sin(theta_arena)*SENSORY_PARAMS['inner_arena_radius'],'-')
  ax.plot(np.cos(theta_arena)*SENSORY_PARAMS['outer_arena_radius'],np.sin(theta_arena)*SENSORY_PARAMS['outer_arena_radius'],'-')
  hpts = []
  for pti in range(nkpwall):
    hpts.append(ax.plot(xwall[pti,t],ywall[pti,t],'o')[0])
  ax.set_aspect('equal')
  ax = plt.subplot(1,2,2)
  ax.plot(distleg.flatten(),wall_touch.flatten(),'k.')
  ax.plot([0,SENSORY_PARAMS['inner_arena_radius'],SENSORY_PARAMS['outer_arena_radius']],
          [SENSORY_PARAMS['arena_height'],SENSORY_PARAMS['arena_height'],0],'-')
  for pti in range(nkpwall):
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
                    xwall_main,ywall_main,
                    xother,yother):

  # increase dimensions if only one frame input
  if xother.ndim < 3:
    xother = xother[:,None]

  npts = xother.shape[0]
  nflies = xother.shape[1]
  T = xother.shape[2]
  
  yother = np.reshape(yother,(npts,nflies,T))
  xeye_main = np.reshape(xeye_main,(1,1,T))
  yeye_main = np.reshape(yeye_main,(1,1,T))
  theta_main = np.reshape(theta_main,(1,1,T))
  
  # don't deal with missing data :)    
  assert(np.any(np.isnan(xeye_main))==False)
  assert(np.any(np.isnan(yeye_main))==False)
  assert(np.any(np.isnan(theta_main))==False)
  
  # vision bin size
  step = 2.*np.pi/SENSORY_PARAMS['n_oma']

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
  tmpbins = np.arange(SENSORY_PARAMS['n_oma'])[:,None,None]
  
  # n_oma x nflies x T
  mindrep = np.tile(mind[None,...],(SENSORY_PARAMS['n_oma'],1,1))
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
  
  otherflies_vision = 1. - np.minimum(1.,SENSORY_PARAMS['otherflies_vision_mult'] * otherflies_vision**SENSORY_PARAMS['otherflies_vision_exp'])

  # t = 249
  # debug_plot_otherflies_vision(t,xother,yother,xeye_main,yeye_main,theta_main,
  #                                 angle0,angle,dist,b_all,otherflies_vision,params)

  # distance from center of arena
  # center of arena is assumed to be [0,0]
  distleg = np.sqrt( xwall_main**2. + ywall_main**2 )

  # height of chamber 
  wall_touch = np.zeros(distleg.shape)
  wall_touch[:] = SENSORY_PARAMS['arena_height']
  wall_touch = np.minimum(SENSORY_PARAMS['arena_height'],np.maximum(0.,SENSORY_PARAMS['arena_height'] - (distleg-SENSORY_PARAMS['inner_arena_radius'])*SENSORY_PARAMS['arena_height']/(SENSORY_PARAMS['outer_arena_radius']-SENSORY_PARAMS['inner_arena_radius'])))
  wall_touch[distleg >= SENSORY_PARAMS['outer_arena_radius']] = 0.
  
  # t = 0
  # debug_plot_wall_touch(t,xlegtip_main,ylegtip_main,distleg,wall_touch,params)

  # to do: add something related to whether the fly is touching another fly
  
  return (otherflies_vision, wall_touch)


featorigin = [mabe.posenames.index('thorax_front_x'),mabe.posenames.index('thorax_front_y')]
feattheta = mabe.posenames.index('orientation')
featglobal = featorigin + [feattheta,]
featthetaglobal = 2
featrelative = np.ones(len(mabe.posenames),dtype=bool)
featrelative[featglobal] = False
nrelative = np.count_nonzero(featrelative)
nglobal = len(featglobal)
nfeatures = len(mabe.posenames)
featangle = np.array([re.search('angle$',s) is not None for s in mabe.posenames])
featangle[feattheta] = True

kpother = [mabe.keypointnames.index('antennae_midpoint'),
            mabe.keypointnames.index('tip_abdomen'),
            mabe.keypointnames.index('left_middle_femur_base'),
            mabe.keypointnames.index('right_middle_femur_base'),
            ]
kpeye = mabe.keypointnames.index('antennae_midpoint')
kpwall = [mabe.keypointnames.index(x) for x in SENSORY_PARAMS['wallkpnames']]
nkpwall = len(kpwall)

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
    i1 = i0 + nkpwall
    res['wall_touch'] = X[...,i0:i1]
  return res

def combine_relative_global(Xrelative,Xglobal):
  X = np.concatenate((Xglobal,Xrelative),axis=-1)
  return X

def compute_pose_features(X,scale):
  posefeat = mabe.kp2feat(X,scale)
  relpose = posefeat[featrelative,...]
  globalpos = posefeat[featglobal,...]

  return relpose,globalpos

def compute_movement(X=None,scale=None,relpose=None,globalpos=None,simplify=None):
  """
  movement = compute_movement(X=X,scale=scale,simplify=simplify_out)
  movement = compute_movement(relpose=relpose,globalpos=globalpos,simplify=simplify_out)

  Args:
      X (ndarray, nkpts x 2 x T x nflies, optional): Keypoints. Can be None only if relpose and globalpos are input. Defaults to None. T>=2
      scale (ndarray, nscale x nflies): Scaling parameters. Can be None only if relpose and globalpos are input. Defaults to None.
      relpose (ndarray, nrelative x T x nflies or nrelative x T, optional): Relative pose features. T>=2
      If input, X and scale are ignored. Defaults to None.
      globalpos (ndarray, nglobal x T x nflies or nglobal x T, optional): Global position. If input, X and scale are ignored. Defaults to None. T>=2
      simplify (string or None, optional): Whether/how to simplify the output. Defaults to None for no simplification.

  Returns:
      movement (ndarray, nfeatures x T-1 x nflies or nfeatures x T-1): Per-frame movement. movement[:,t,i] is the movement from frame 
      t to t+1 for fly i. 
  """

  if relpose is None or globalpos is None:
    relpose,globalpos = compute_pose_features(X,scale)
    
  nd = relpose.ndim
  assert(nd==2 or nd==3)
  if nd < 3:
    relpose = relpose[...,None]
    globalpos = globalpos[...,None]
  T = relpose.shape[1]
  nflies = relpose.shape[2]

  Xorigin = globalpos[:2,...]
  Xtheta = globalpos[2,...]  
  dXoriginrel = mabe.rotate_2d_points((Xorigin[:,1:,:]-Xorigin[:,:-1,:]).transpose((1,0,2)),Xtheta[:-1,:]).transpose((1,0,2))
  forward_vel = dXoriginrel[1,...]
  sideways_vel = dXoriginrel[0,...]

  movement = np.zeros([nfeatures,T-1,nflies],relpose.dtype)
  movement[featrelative,...] = relpose[:,1:,:]-relpose[:,:-1,:]
  movement[featorigin[0],...] = forward_vel
  movement[featorigin[1],...] = sideways_vel
  movement[feattheta,...] = Xtheta[1:,...]-Xtheta[:-1,...]
  movement[featangle,...] = mabe.modrange(movement[featangle,...],-np.pi,np.pi)

  if simplify is not None:
    if simplify == 'global':
      movement = movement[featglobal,...]
    else:
      raise

  if nd == 2:
    movement = movement[...,0]

  return movement


def compute_sensory_wrapper(Xkp,flynum,theta_main=None,returnall=False):
  
  # other flies positions
  idxother = np.ones(Xkp.shape[-1],dtype=bool)
  idxother[flynum] = False
  Xkp_other = Xkp[:,:,:,idxother]
  
  xeye_main = Xkp[kpeye,0,:,flynum]
  yeye_main = Xkp[kpeye,1,:,flynum]
  xwall_main = Xkp[kpwall,0,:,flynum]
  ywall_main = Xkp[kpwall,1,:,flynum]
  xother = Xkp_other[kpother,0,...].transpose((0,2,1))
  yother = Xkp_other[kpother,1,...].transpose((0,2,1))
  
  if theta_main is None:
    _,_,theta_main = mabe.body_centric_kp(Xkp[...,[flynum,]])
    theta_main = theta_main[...,0]
  
  otherflies_vision,wall_touch = \
    compute_sensory(xeye_main,yeye_main,theta_main+np.pi/2,
                    xwall_main,ywall_main,
                    xother,yother)
  sensory = np.r_[wall_touch,otherflies_vision]
  if returnall:
    return sensory,wall_touch,otherflies_vision
  else:
    return sensory

def combine_inputs(relpose=None,sensory=None,input=None,labels=None,dim=0):
  if input is None:
    input = np.concatenate((relpose,sensory),axis=dim)
  if labels is not None:
    input = np.concatenate((input,labels),axis=dim)
  return input 


def compute_features(X,id,flynum,scale_perfly,smush=True,outtype=None,
                         simplify_out=None,simplify_in=None):
  
  res = {}
  
  # convert to relative locations of body parts
  if id is None:
    scale = scale_perfly
  else:
    scale = scale_perfly[:,id]
  
  relpose,globalpos = compute_pose_features(X[...,flynum],scale)
  relpose = relpose[...,0]
  globalpos = globalpos[...,0]
  sensory,wall_touch,otherflies_vision = \
    compute_sensory_wrapper(X,flynum,theta_main=globalpos[featthetaglobal,...],
                            returnall=True)
  input = combine_inputs(relpose=relpose,sensory=sensory).T
  res['input'] = input[:-1,:]

  movement = compute_movement(relpose=relpose,globalpos=globalpos,simplify=simplify_out)
  if simplify_out is not None:
    if simplify_out == 'global':
      movement = movement[featglobal,...]
    else:
      raise

  res['labels'] = movement.T
  res['init'] = globalpos[:,:2]
  res['scale'] = scale
  res['nextinput'] = input[-1,:]
  
  if not smush:
    res['global'] = globalpos[:,:-1]
    res['relative'] = relpose[:,:-1]
    res['nextglobal'] = globalpos[:,-1]
    res['nextrelative'] = relpose[:,-1]
    if simplify_in == 'no_sensory':
      pass
    else:
      res['wall_touch'] = wall_touch[:,:-1]
      res['otherflies_vision'] = otherflies_vision[:,:-1]
      res['next_wall_touch'] = wall_touch[:,-1]
      res['next_otherflies_vision'] = otherflies_vision[:,-1]
    
  # debug_plot_compute_features(X,porigin,theta,Xother,Xnother)
    
  if outtype is not None:
    res = {key: val.astype(outtype) for key,val in res.items()}
  return res


# def compute_features(X,id,flynum,scale_perfly,sensory_params,smush=True,outtype=None,
#                      simplify_out=None,simplify_in=None):
  
#   res = {}
  
#   # convert to relative locations of body parts
#   if id is None:
#     scale = scale_perfly
#   else:
#     scale = scale_perfly[:,id]
#   Xfeat = mabe.kp2feat(X[...,flynum],scale)
#   Xfeat = Xfeat[...,0]
  
#   # compute global coordinates relative to previous frame
#   Xorigin = Xfeat[featorigin,...]
#   Xtheta = Xfeat[feattheta,...]
#   dXoriginrel = mabe.rotate_2d_points((Xorigin[:,1:]-Xorigin[:,:-1]).T,Xtheta[:-1]).T
#   forward_vel = dXoriginrel[1,:]
#   sideways_vel = dXoriginrel[0,:]

#   # compute sensory information

#   if simplify_in == 'no_sensory':
#     input = Xfeat[featrelative,:].T
#     res['input'] = input[:-1,:]
#   else:
#     # other flies positions
#     idxother = np.ones(X.shape[-1],dtype=bool)
#     idxother[flynum] = False
#     Xother = X[:,:,:,idxother]
    
#     xeye_main = X[kpeye,0,:,flynum]
#     yeye_main = X[kpeye,1,:,flynum]
#     xlegtip_main = X[kplegtip,0,:,flynum]
#     ylegtip_main = X[kplegtip,1,:,flynum]
#     xother = Xother[kpother,0,...].transpose((0,2,1))
#     yother = Xother[kpother,1,...].transpose((0,2,1))
#     theta_main = Xfeat[feattheta,...]+np.pi/2
    
#     otherflies_vision,wall_touch = \
#       compute_sensory(xeye_main,yeye_main,theta_main,
#                       xlegtip_main,ylegtip_main,
#                       xother,yother,sensory_params)
#     input = np.r_[Xfeat[featrelative,:],wall_touch[:,:],otherflies_vision[:,:]].T
#     res['input'] = input[:-1,:]
    
#   movement = Xfeat[:,1:]-Xfeat[:,:-1]
#   movement[featorigin[0],:] = forward_vel
#   movement[featorigin[1],:] = sideways_vel
#   movement[featangle,...] = mabe.modrange(movement[featangle,...],-np.pi,np.pi)
#   if simplify_out is not None:
#     if simplify_out == 'global':
#       movement = movement[featglobal,...]
#     else:
#       raise

#   res['labels'] = movement.T
#   res['init'] = Xfeat[featrelative==False,:2]
#   res['scale'] = scale_perfly[:,id]
#   res['nextinput'] = input[-1,:]
  
#   # # check that we can undo correctly
#   # res['labels'].T
#   # thetavel = res['labels'][:,2]
#   # theta0 = res['init'][2,0]
#   # thetare = np.cumsum(np.r_[theta0[None],thetavel],axis=0)
#   # np.max(np.abs(thetare-Xfeat[2,:]))
#   # doriginrel = res['labels'][:,[1,0]]
#   # dxy = mabe.rotate_2d_points(doriginrel,-thetare[:-1])
#   # Xorigin[:,1:]-Xorigin[:,:-1]
#   # np.max(np.sum(np.abs((Xorigin[:,1:]-Xorigin[:,:-1]).T-dxy),axis=0))
  
#   if not smush:
#     res['global'] = Xfeat[featrelative==False,:-1]
#     res['relative'] = Xfeat[featrelative,:-1]
#     res['nextglobal'] = Xfeat[featrelative==False,-1]
#     res['nextrelative'] = Xfeat[featrelative,-1]
#     if simplify_in == 'no_sensory':
#       pass
#     else:
#       res['wall_touch'] = wall_touch[:,:-1]
#       res['otherflies_vision'] = otherflies_vision[:,:-1]
#       res['next_wall_touch'] = wall_touch[:,-1]
#       res['next_otherflies_vision'] = otherflies_vision[:,-1]
    
#   # debug_plot_compute_features(X,porigin,theta,Xother,Xnother)
    
#   if outtype is not None:
#     res = {key: val.astype(outtype) for key,val in res.items()}
#   return res
    
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

def init_train_bpe(zlabels,transform=True,max_val=10,
                   n_clusters=int(1e3)):  

  def apply_transform(z):
    x = np.zeros(z.shape)
    x = np.sqrt(np.minimum(max_val,np.abs(z)))*np.sign(z)
    return x
  
  def apply_inverse_transform(x):
    z = np.zeros(x.shape)
    z = x**2*np.sign(z)
    return z

  # for |x| <= transform_thresh, use sqrt. above, use log
  if transform:
    x = apply_transform(zlabels)
  else:
    x = zlabels.copy()

  # k-means clustering of inter-frame motion
  alg = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters)
  token = alg.fit_predict(x)
  centers = alg.cluster_centers_  
  err = np.abs(x - centers[token,:])
  
  cmap = plt.get_cmap("rainbow")
  colors = cmap(np.arange(n_clusters)/n_clusters)
  colors = colors[np.random.permutation(n_clusters),:]*.7
  
  nplot = 1000
  nstd = 1
  fig,ax = plt.subplots(2,1,sharex='all')
  ax[0].cla()
  ax[1].cla()
  for i in range(x.shape[1]):
    xrecon = centers[token[:nplot],i]
    ax[0].plot([0,nplot],[i*nstd,]*2,':',color=[.5,.5,.5])
    ax[0].plot(np.arange(nplot),i*nstd*2+x[:nplot,i],'k.-')
    tmpy = np.c_[xrecon,x[:nplot,i],np.zeros(nplot)]
    tmpy[:,2] = np.nan
    tmpx = np.tile(np.arange(nplot)[:,None],(1,3))
    ax[0].plot(tmpx.flatten(),i*nstd*2+tmpy.flatten(),'k-')
    ax[0].scatter(np.arange(nplot),i*nstd*2+xrecon,c=colors[token[:nplot],:],marker='o')
    ax[0].text(0,2*nstd*(i+.5),outnames[i],horizontalalignment='left',verticalalignment='top')

  ax[1].plot(np.arange(nplot),token[:nplot],'k-')
  ax[1].scatter(np.arange(nplot),token[:nplot],c=colors[token[:nplot],:],marker='o')
  ax[1].set_ylabel('Token ID')
  return

def train_bpe(data,scale_perfly,simplify_out=None):
  
  # collect motion data
  nflies = data['X'].shape[3]
  T = data['X'].shape[2]

  isdata = np.any(np.isnan(data['X']),axis=(0,1)) == False
  isstart = (data['ids'][1:,:]!=data['ids'][:-1,:]) | \
    (data['frames'][1:,:] != (data['frames'][:-1,:]+1))
  isstart = np.r_[np.ones((1,nflies),dtype=bool),isstart]
  labels = None

  print('Computing movements over all data')
  for i in tqdm.trange(nflies,desc='animal'):
    isstart = isdata[1:,i] & \
      ((isdata[:-1,i] == False) | \
       (data['ids'][1:,i]!=data['ids'][:-1,i]) | \
       (data['frames'][1:,0] != (data['frames'][:-1,0]+1)))
    isstart = np.r_[isdata[0,i],isstart]
    isend = isdata[:-1,i] & \
      ((isdata[1:,i]==False) | \
       (data['ids'][1:,i]!=data['ids'][:-1,i]) | \
       (data['frames'][1:,0] != (data['frames'][:-1,0]+1)))
    isend = np.r_[isend,isdata[-1,i]]
    t0s = np.nonzero(isstart)[0]
    t1s = np.nonzero(isend)[0]+1
    for j in tqdm.trange(len(t0s),desc='frames'):
      t0 = t0s[j]
      t1 = t1s[j]
      id = data['ids'][t0,i]
      xcurr = compute_features(data['X'][:,:,t0:t1,:],id,i,scale_perfly,None,
                               simplify_in='no_sensory')
                               #simplify_out=simplify_out)
      labelscurr = xcurr['labels']
      if labels is None:
        labels = labelscurr
      else:
        labels = np.r_[labels,labelscurr]
        
        
    # zscore
    mu = np.mean(labels,axis=0)
    sig = np.std(labels,axis=0)
    zlabels = (labels-mu)/sig
    
    
    
  return

def apply_mask(x,mask,nin=0,maskflagged=False):
  # mask with zeros
  if maskflagged:
    if mask is not None:
      x[mask,:-nin-1] = 0.
      x[mask,-1] = 1.    
  else:
    if mask is None:
      mask = torch.zeros(x.shape[:-1],dtype=x.dtype)
    else:
      x[mask,:-nin] = 0.
    x = torch.cat((x,mask[...,None].type(x.dtype)),dim=-1) 
  return x

def unzscore(x,mu,sig):
  return x*sig + mu
def zscore(x,mu,sig):
  return (x-mu)/sig

class FlyMLMDataset(torch.utils.data.Dataset):
  def __init__(self,data,max_mask_length=None,pmask=None,masktype='block',
               simplify_out=None,simplify_in=None,pdropout_past=0.,maskflag=None):
    self.data = data
    # number of outputs
    self.d_output = self.data[0]['labels'].shape[-1]
    # number of inputs
    self.dfeat = self.data[0]['input'].shape[-1]

    self.max_mask_length = max_mask_length
    self.pmask = pmask
    self.masktype = masktype
    self.pdropout_past = pdropout_past
    self.simplify_out = simplify_out # modulation of task to make it easier
    self.simplify_in = simplify_in
    if maskflag is None:
      maskflag = (masktype is not None) or (pdropout_past>0.)
    self.maskflag = maskflag
    
    self.mu_input = None
    self.sig_input = None
    self.mu_labels = None
    self.sig_labels = None
    
    self.dtype = np.float32
    
  def hasmaskflag(self):
    return self.ismasked() or self.maskflag or self.pdropout_past > 0
    
  def ismasked(self):
    """Whether this object is a dataset for a masked language model, ow a causal model.
    v = self.ismasked()

    Returns:
        bool: Whether data are masked. 
    """
    return self.masktype is not None
    
  def zscore(self,mu_input=None,sig_input=None,mu_labels=None,sig_labels=None):
    """
    self.zscore(mu_input=None,sig_input=None,mu_labels=None,sig_labels=None)
    zscore the data. input and labels are z-scored for each example in self.data
    and converted to float32. They are stored in place in the dict for each example
    in the dataset. If mean and standard deviation statistics are input, then
    these statistics are used for z-scoring. Otherwise, means and standard deviations
    are computed from this data. 

    Args:
        mu_input (ndarray, dfeat, optional): Pre-computed mean for z-scoring input. 
        If None, mu_input is computed as the mean of all the inputs in self.data. 
        Defaults to None.
        sig_input (ndarray, dfeat, optional): Pre-computed standard deviation for 
        z-scoring input. If mu_input is None, sig_input is computed as the std of all 
        the inputs in self.data. Defaults to None. Do not set this to None if mu_input 
        is not None. 
        mu_labels (ndarray, d_output, optional): Pre-computed mean for z-scoring labels. 
        If None, mu_labels is computed as the mean of all the labels in self.data. 
        Defaults to None.
        sig_labels (ndarray, dfeat, optional): Pre-computed standard deviation for 
        z-scoring labels. If mu_labels is None, sig_labels is computed as the standard 
        deviation of all the labels in self.data. Defaults to None. Do not set this 
        to None if mu_labels is not None. 
        
    No value returned. 
    """
    
    def zscore_helper(f):
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
      self.mu_input,self.sig_input = zscore_helper('input')
    else:
      self.mu_input = mu_input.copy()
      self.sig_input = sig_input.copy()

    self.mu_input = self.mu_input.astype(self.dtype)
    self.sig_input = self.sig_input.astype(self.dtype)
      
    if mu_labels is None:
      self.mu_labels,self.sig_labels = zscore_helper('labels')
    else:
      self.mu_labels = mu_labels.copy()
      self.sig_labels = sig_labels.copy()      

    self.mu_labels = self.mu_labels.astype(self.dtype)
    self.sig_labels = self.sig_labels.astype(self.dtype)
    
    for example in self.data:
      example['input'] = self.zscore_input(example['input'])
      example['labels'] = self.zscore_labels(example['labels'])
      
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
  
  def maskind(self,inl,pmask=None):
    if pmask is None:
      pmask = self.pmask
    mask = torch.rand(inl)<=pmask
    if not torch.any(mask):
      imask = np.random.randint(inl)
      mask[imask] = True
    return mask
  
  def set_masktype(self,masktype):
    self.masktype = masktype
    
  def zscore_input(self,rawinput):
    if self.mu_input is None:
      input = rawinput.copy()
    else:
      input = (rawinput-self.mu_input)/self.sig_input
    return input.astype(self.dtype)
  
  def zscore_labels(self,rawlabels):
    if self.mu_labels is None:
      labels = rawlabels.copy()
    else:
      labels = (rawlabels-self.mu_labels)/self.sig_labels
    return labels.astype(self.dtype)
  
  def mask_input(self,input,masktype='default'):

    if masktype == 'default':
      masktype = self.masktype
    
    contextl = input.shape[0]
    
    if self.masktype == 'block':
      mask = self.maskblock(contextl)
    elif self.masktype == 'ind':
      mask = self.maskind(contextl)
    elif self.masktype == 'last':
      mask = self.masklast(contextl)
    else:
      mask = None      
    maskflagged = False
    if self.masktype is not None:
      input = apply_mask(input,mask,self.dfeat)
      maskflagged = True
    if self.pdropout_past > 0:
      dropout_mask = self.maskind(contextl,pmask=self.pdropout_past)
      input = apply_mask(input,dropout_mask,self.dfeat,maskflagged)
      maskflagged = True
    else:
      dropout_mask = None
    if self.maskflag and not maskflagged:
      input = apply_mask(input,None)
    
    return input,mask,dropout_mask
    
  def input2pose(self,input):
    """
    pose = self.input2pose(input)
    Extracts the relative pose features. 

    Args:
        input (ndarray or Tensor, (... x nfeat) or (... x (d_output+nfeat+ismasked)): input features
        to process. 

    Returns:
        pose ndarray or Tensor, ... x npose: relative pose features extracted from input
    """
    if input.shape[-1] > self.dfeat:
      if self.hasmaskflag():
        input = input[...,self.d_output:-1]
      else:
        input = input[...,self.d_output:]
    res = split_features(input,simplify=self.simplify_in)
    return res['pose']
    
  def __getitem__(self,idx: int):
    """
    example = self.getitem(idx)
    Returns dataset example idx. It performs the following processing:
    - Converts the data to tensors.
    - Concatenates labels and feature input into input, and shifts labels in inputs
    depending on whether this is a dataset for a masked LM or a causal LM (see below). 
    - For masked LMs, draw and applies a random mask of type self.masktype. 

    Args:
        idx (int): Index of the example to return. 

    Returns:
        example (dict) with the following fields:

        For masked LMs: 
        example['input'] is a tensor of shape contextl x (d_output + dfeat + 1)
        where example['input'][t,:d_output] is the motion from frame t to t+1 and
        example['input'][t,d_output:-1] are the input features for frame t. 
        example['input'][t,-1] indicates whether the frame is masked or not. If this 
        frame is masked, then example['input'][t,:d_output] will be set to 0. 
        example['labels'] is a tensor of shape contextl x d_output
        where example['labels'][t,:] is the motion from frame t to t+1. 
        example['init'] is a tensor of shape dglobal, corresponding to the global
        position in frame 0. 
        example['mask'] is a tensor of shape contextl indicating which frames are masked.
        
        For causal LMs:
        example['input'] is a tensor of shape (contextl-1) x (d_output + dfeat)
        where example['input'][t,:d_output] is the motion from frame t to t+1 and
        example['input'][t,d_output:] are the input features for frame t+1. 
        example['labels'] is a tensor of shape contextl x d_output
        where example['labels'][t,:] is the motion from frame t+1 to t+2.
        example['init'] is a tensor of shape dglobal, corresponding to the global
        position in frame 1. 

        For all:
        example['scale'] are the scale features for this fly, used for converting from
        relative pose features to keypoints. 
        example['categories'] are the currently unused categories for this sequence.
        example['metadata'] is a dict of metadata about this sequence.
        
    """
    input = torch.as_tensor(self.data[idx]['input'])
    labels = torch.as_tensor(self.data[idx]['labels'].copy())
    nin = input.shape[-1]
    contextl = input.shape[0]
    if self.ismasked():
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
    input,mask,dropout_mask = self.mask_input(input)
    res = {'input': input, 'labels': labels, 
            'init': init, 'scale': scale, 'categories': categories,
            'metadata': self.data[idx]['metadata'].copy()}
    if self.masktype is not None:
      res['mask'] = mask
    if dropout_mask is not None:
      res['dropout_mask'] = dropout_mask
    return res
  
  def __len__(self):
    return len(self.data)
  
  def pred2pose(self,posein,globalin,pred,isnorm=True):
    """
    posenext,globalnext = self.pred2pose(posein,globalin,pred)
    Adds the motion in pred to the pose defined by posein and globalin 
    to compute the pose in the next frame. 

    Args:
        posein (ndarray, dposerel): Unnormalized relative pose features for the current 
        frame globalin (ndarray, dglobal): Global position for the current frame
        pred (ndarray, d_output): Z-scored (if applicable) predicted movement from 
        current frame to the next frame
        isnorm (bool,optional): Whether the input pred is normalized. Default: True.

    Returns:
        posenext (ndarray, dposerel): Unnormalized relative pose features for the next 
        frame. 
        globalnext (ndarray, dglobal): Global position for the next frame. 
        
    """
    if self.mu_labels is not None and isnorm:
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
  
  def get_Xfeat(self,input0=None,global0=None,movements=None,example=None):
    """
    Xfeat = self.get_Xfeat(input0,global0,movements)

    Unnormalizes initial input input0 and extracts relative pose features. Combines
    these with global0 to get the full set of pose features for initial frame 0. 
    Converts egocentric movements (forward, sideway) to global, and computes the
    full pose features for each frame based on the input movements. 

    Either input0, global0, and movements must be input OR 
    example must be input, and input0, global0, and movements are derived from there.

    Args:
        input0 (ndarray, d_output+dfeat+hasmaskflag): network input for time point 0
        global0 (ndarray, 3): global position at time point 0
        movements (ndarray, T x d_output ): movements[t,:] is the movement from t to t+1

    Returns:
        Xfeat: (ndarray, T+1 x nfeatures): All pose features for frames 0 through T
    """
    
    if example is not None:
      if input0 is None:
        input0 = example['input']
      if global0 is None:
        global0 = example['init']
      if movements is None:
        movements = example['labels']

    if torch.is_tensor(input0):
      input0 = input0.numpy()
    if torch.is_tensor(global0):
      global0 = global0.numpy()
    if torch.is_tensor(movements):
      movements = movements.numpy()
    
    nlabel = movements.shape[-1]
    if self.hasmaskflag():
      input0 = input0[...,:-1]
    input0 = input0[...,nlabel:]
    if self.mu_input is not None:
      input0 = unzscore(input0,self.mu_input,self.sig_input)
      movements = unzscore(movements,self.mu_labels,self.sig_labels)
      
    input0 = split_features(input0,simplify=self.simplify_in)
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
      Xfeat[:,featrelative] = np.tile(input0['pose'],(movements.shape[0]+1,1))
    else:
      Xfeatpose = np.cumsum(np.r_[input0['pose'][None,:],movements[:,featrelative]],axis=0)
      Xfeat[:,featrelative] = Xfeatpose
    
    return Xfeat
  
  def get_Xkp(self,input0=None,global0=None,movements=None,scale=None,example=None):
    """
    Xkp = self.get_Xkp(input0,global0,movements)

    Call get_Xfeat to get the full pose features based on the initial input and global
    position input0 and global0 and the per-frame motion in movements. Converts
    the full pose features to keypoint coordinates. 
    
    Either input0, global0, movements, and scale must be input OR 
    example must be input, and input0, global0, movements, and scale are derived from there

    Args:
        input0 (ndarray, d_output+dfeat+hasmaskflag): network input for time point 0
        global0 (ndarray, 3): global position at time point 0
        movements (ndarray, T x d_output ): movements[t,:] is the movement from t to t+1
        scale (ndarray, dscale): scale parameters for this fly
        example (dict), output of __getitem__: example with fields input, init, labels, and
        scale. 

    Returns:
        Xkp: (ndarray, nkeypoints x 2 x T+1 x 1): Keypoint locations for frames 0 through T
    """
    
    if example is not None and scale is None:
      scale = example['scale']
    if torch.is_tensor(scale):
      scale = scale.numpy()
    
    Xfeat = self.get_Xfeat(input0=input0,global0=global0,movements=movements,example=example)
    Xkp = self.feat2kp(Xfeat,scale)
    return Xkp
  
  def feat2kp(self,Xfeat,scale):
    """
    Xkp = self.feat2kp(Xfeat)

    Args:
        Xfeat (ndarray, T x nfeatures): full pose features for each frame
        scale (ndarray, dscale): scale features

    Returns:
        Xkp (ndarray, nkeypoints x 2 x T+1 x 1): keypoints for each frame
    """
    Xkp = mabe.feat2kp(Xfeat.T[...,None],scale[...,None])
    return Xkp
    
  def get_outnames(self):
    """
    outnames = self.get_outnames()

    Returns:
        outnames (list of strings): names of each output motion
    """
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

def prob_causal_criterion(tgt,pred):
  d = tgt.shape[-1]
  err = torch.sum(pred['stateprob']*torch.sum(torch.abs(tgt[...,None]-pred['perstate'])/d,keepdim=False,axis=-2))
  return err

def min_causal_criterion(tgt,pred):
  d = tgt.shape[-1]
  errperstate = torch.sum(torch.abs(tgt[...,None]-pred)/d,keepdim=False,dim=tuple(range(pred.ndim - 1)))
  err = torch.min(errperstate,dim=-1)
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

class TransformerBestStateModel(torch.nn.Module):

  def __init__(self, d_input: int, d_output: int,
               d_model: int = 2048, nhead: int = 8, d_hid: int = 512,
               nlayers: int = 12, dropout: float = 0.1, nstates: int = 8):
    super().__init__()
    self.model_type = 'TransformerBestState'

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

    # for each hidden state, fully connected layer from model to output size
    # concatenated together, so output is size d_output * nstates
    self.decode = torch.nn.Linear(d_model,nstates*d_output)

    # store hyperparameters
    self.d_model = d_model
    self.nstates = nstates
    self.d_output = d_output

    self.init_weights()

  def init_weights(self) -> None:
    pass

  def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
      src: Tensor, shape [batch_size,seq_len,dinput]
      src_mask: Tensor, shape [seq_len,seq_len]
    Returns:
      Tensor of shape [batch_size, seq_len, d_output, nstates]
    """

    # project input into d_model space, multiple by sqrt(d_model) for reasons?
    src = self.encoder(src) * math.sqrt(self.d_model)

    # add in the positional encoding of where in the sentence the words occur
    # it is weird to me that these are added, but I guess it would be almost
    # the same to have these be combined in a single linear layer
    src = self.pos_encoder(src)

    # main transformer layers
    transformer_output = self.transformer_encoder(src,src_mask)

    # output given each hidden state  
    # batch_size x seq_len x d_output x nstates
    output = self.decode(transformer_output).reshape(src.shape[:-1]+(self.d_output,self.nstates))
      
    return output

  def randpred(self,pred):
    contextl = pred.shape[-3]
    draw = torch.randint(0,pred.shape[-1],contextl)
    return pred[...,np.arange(contextl,dtype=int),:,draw]

class TransformerStateModel(torch.nn.Module):

  def __init__(self, d_input: int, d_output: int,
               d_model: int = 2048, nhead: int = 8, d_hid: int = 512,
               nlayers: int = 12, dropout: float = 0.1, nstates: int = 64,
               minstateprob: float = None):
    super().__init__()
    self.model_type = 'TransformerState'

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

    # from output of transformer layers to hidden state probabilities
    self.state = torch.nn.Sequential(
      torch.nn.Linear(d_model,nstates),
      torch.nn.Dropout(dropout),
      torch.nn.Softmax(dim=-1)
    )
    if minstateprob is None:
      minstateprob = .01/nstates
    # for each hidden state, fully connected layer from model to output size
    # concatenated together, so output is size d_output * nstates
    self.decode = torch.nn.Linear(d_model,nstates*d_output)

    # store hyperparameters
    self.d_model = d_model
    self.nstates = nstates
    self.d_output = d_output
    self.minstateprob = minstateprob

    self.init_weights()

  def init_weights(self) -> None:
    pass

  def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
      src: Tensor, shape [batch_size,seq_len,dinput]
      src_mask: Tensor, shape [seq_len,seq_len]
    Returns:
      output dict with the following fields:
      stateprob: Tensor of shape [batch_size, seq_len, nstates] indicating the 
      probability of each state
      perstate: Tensor of shape [batch_size, seq_len, d_output, nstates] where
      perstate[t,i,:,j] is the output for time t, example i, and state j. 
    """

    # project input into d_model space, multiple by sqrt(d_model) for reasons?
    src = self.encoder(src) * math.sqrt(self.d_model)

    # add in the positional encoding of where in the sentence the words occur
    # it is weird to me that these are added, but I guess it would be almost
    # the same to have these be combined in a single linear layer
    src = self.pos_encoder(src)

    # main transformer layers
    transformer_output = self.transformer_encoder(src,src_mask)

    output = {}
    # probability of each of the hidden states
    # batch_size x seq_len x nstates
    output['stateprob'] = self.state(transformer_output)

    # make sure that every state has some probability
    if self.training:
      output['stateprob'] = (output['stateprob']+self.minstateprob)/(1+self.nstates*self.minstateprob)
      
    # output given each hidden state  
    # batch_size x seq_len x d_output x nstates
    output['perstate'] = self.decode(transformer_output).reshape(src.shape[:-1]+(self.d_output,self.nstates))
      
    return output
  
  def maxpred(self,pred):
    state = torch.argmax(pred['stateprob'],axis=-1)
    perstate = pred['perstate'].flatten(end_dim=1)
    out = perstate[torch.arange(perstate.shape[0],dtype=int),:,state.flatten()].reshape(pred['perstate'].shape[:-1])
    return out
  
  def randpred(self,pred):
    state = torch.multinomial(pred['stateprob'].flatten(end_dim=-2),1)
    perstate = pred['perstate'].flatten(end_dim=1)
    out = perstate[torch.arange(perstate.shape[0],dtype=int),:,state.flatten()].reshape(pred['perstate'].shape[:-1])
    return out
  
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

def predict_open_loop(Xkp,fliespred,scales,burnin,dataset,model,maxcontextl=np.inf,debug=False):
# def predict_open_loop(example0,Xkp_others,scale,dataset,tpred,
#                       sensory_params,movementtrue=None,maxcontextl=None):
  """
  Xkp = predict_open_loop(Xkp,fliespred,scales,burnin,dataset,model,sensory_params,maxcontextl=np.inf,debug=False)

  Args:
    Xkp (ndarray, nkpts x 2 x tpred x nflies): keypoints for all flies for all frames.
    Can be nan for frames/flies to be predicted. Will be overwritten. 
    fliespred (ndarray, nfliespred): indices of flies to predict
    scales (ndarray, nscale x nfliespred): scale parameters for the flies to be predicted
    burnin (int): number of frames to use for initialization
    dataset (FLYMLMDataset): dataset is used for determining input/output processing
    maxcontextl (int, optional): maximum number of frames to use for context. Default np.inf
    debug (bool, optional): whether to fill in from movement computed from Xkp_all

  Returns:
    Xkp (ndarray, nkpts x 2 x tpred x nflies): keypoints for all flies for all frames,
    with predicted frames/flies filled in. 
  """
  model.eval()

  with torch.no_grad():
    w = next(iter(model.parameters()))
    dtype = w.cpu().numpy().dtype
    device = w.device

  tpred = Xkp.shape[-2]
  nfliespred = len(fliespred)
  relpose = np.zeros((tpred,nrelative,nfliespred),dtype=dtype)
  globalpos = np.zeros((tpred,nglobal,nfliespred),dtype=dtype)
  zmovement = np.zeros((tpred-1,nfeatures,nfliespred),dtype=dtype)
  sensory = None
  zinputs = None

  if debug:
    movement_true = compute_movement(X=Xkp[...,fliespred],scale=scales,simplify=dataset.simplify_out).transpose((1,0,2)).astype(dtype)
  
  # outputs -- hide frames past burnin
  Xkp[:,:,burnin:,fliespred] = np.nan
  
  # compute the pose for pred flies for first burnin frames
  relpose0,globalpos0 = compute_pose_features(Xkp[...,:burnin,fliespred],scales)
  relpose[:burnin,:,:] = relpose0.transpose((1,0,2))
  globalpos[:burnin,:,:] = globalpos0.transpose((1,0,2))
  # compute movement for pred flies between first burnin frames
  movement0 = compute_movement(relpose=relpose0,
                               globalpos=globalpos0,
                               simplify=dataset.simplify_out)
  movement0 = movement0.transpose((1,0,2))
  for i in range(nfliespred):
    zmovementcurr = dataset.zscore_labels(movement0[...,i])
    zmovement[:burnin-1,:,i] = zmovementcurr
    
  # compute sensory features for first burnin frames
  if dataset.simplify_in is None:
    for i in range(nfliespred):
      flynum = fliespred[i]
      sensorycurr = compute_sensory_wrapper(Xkp[...,:burnin,:],flynum,
                                            theta_main=globalpos[:burnin,featthetaglobal,i])
      if sensory is None:
        nsensory = sensorycurr.shape[0]
        sensory = np.zeros((tpred,nsensory,nfliespred),dtype=dtype)
      sensory[:burnin,:,i] = sensorycurr.T
 
  for i in range(nfliespred):
    if dataset.simplify_in is None:
      rawinputscurr = combine_inputs(relpose=relpose[:burnin,:,i],
                                  sensory=sensory[:burnin,:,i],dim=1)
    else:
      rawinputscurr = relpose[:burnin,:,i]
      
    zinputscurr = dataset.zscore_input(rawinputscurr)    
    if zinputs is None:
      ninput = zinputscurr.shape[1]
      zinputs = np.zeros((tpred,ninput,nfliespred),dtype=dtype)
    zinputs[:burnin,:,i] = zinputscurr
    
  
  if dataset.ismasked():
    masktype = 'last'
    dummy = np.zeros((1,example['labels'].shape[-1]))
    dummy[:] = np.nan
  else:
    masktype = None
  
  # start predicting motion from frame burnin-1 to burnin = t
  for t in tqdm.trange(burnin,tpred):
    t0 = int(np.maximum(t-maxcontextl,0))
    
    if dataset.ismasked():
      net_mask = generate_square_full_mask(t-t0-1).to(device)
    else:
      net_mask = generate_square_subsequent_mask(t-t0-1).to(device)
    
    for i in range(nfliespred):
      flynum = fliespred[i]
      zinputcurr = zinputs[t0:t,:,i]
      relposecurr = relpose[t-1,:,i]
      globalposcurr = globalpos[t-1,:,i]
      zmovementin = zmovement[t0:t-1,:,i]
      if dataset.ismasked():
        zmovementin = np.r_[zmovementin,dummy]
      else:
        zinputcurr = zinputcurr[1:,...]
      xcurr = torch.tensor(np.concatenate((zmovementin,zinputcurr),axis=-1))
      xcurr,_,_ = dataset.mask_input(xcurr,masktype)
      
      if debug:
        zmovementout = dataset.zscore_labels(movement_true[t-1,:,i]).astype(dtype)
      else:
        with torch.no_grad():
          # predict for all frames
          # masked: movement from 0->1, ..., t->t+1
          # causal: movement from 1->2, ..., t->t+1
          # last prediction: t->t+1
          pred = model(xcurr[None,...].to(device),net_mask)
          if model.model_type == 'TransformerBestState' or model.model_type == 'TransformerState':
            pred = model.randpred(pred)
          # z-scored movement from t to t+1
          pred = pred[0,-1,:].cpu()
        zmovementout = pred.numpy()
      relposenext,globalposnext = dataset.pred2pose(relposecurr,globalposcurr,zmovementout)
      relpose[t,:,i] = relposenext
      globalpos[t,:,i] = globalposnext
      zmovement[t-1,:,i] = zmovementout
      featnext = combine_relative_global(relposenext,globalposnext)
      Xkp_next = mabe.feat2kp(featnext,scales[...,i])
      Xkp_next = Xkp_next[:,:,0,0]
      Xkp[:,:,t,flynum] = Xkp_next

    if dataset.simplify_in is None:
      for i in range(nfliespred):
        flynum = fliespred[i]
        sensorynext = compute_sensory_wrapper(Xkp[...,[t,],:],flynum,
                                              theta_main=globalpos[[t,],featthetaglobal,i])
        sensory[t,:,i] = sensorynext.T
 
    for i in range(nfliespred):
      if dataset.simplify_in is None:
        rawinputsnext = combine_inputs(relpose=relpose[[t,],:,i],
                                       sensory=sensory[[t,],:,i],dim=1)
      else:
        rawinputsnext = relpose[[t,],:,i]
      zinputsnext = dataset.zscore_input(rawinputsnext)
      zinputs[t,:,i] = zinputsnext

  return Xkp

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

def debug_plot_batch_state(stateprob,nsamplesplot=3,
                          h=None,ax=None,fig=None):
  batch_size = stateprob.shape[0]

  samplesplot = np.round(np.linspace(0,batch_size-1,nsamplesplot)).astype(int)

  if ax is None:
    fig,ax = plt.subplots(nsamplesplot,1)
  if h is None:
    h = [None,]*nsamplesplot

  for i in range(nsamplesplot):
    iplot = samplesplot[i]
    if h[i] is None:
      h[i] = ax[i].imshow(stateprob[iplot,:,:].T,vmin=0.,vmax=1.)
    else:
      h[i].set_data(stateprob[iplot,:,:].T)
    ax[i].axis('auto')
  
  fig.tight_layout(h_pad=0)
  return h,ax,fig

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
      zmovement_pred = None
    else:
      Xcenter_pred = None
      Xtheta_pred = None
      zmovement_pred = None
    
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

def debug_plot_pose_prob(example,train_dataset,predcpu,tplot,fig=None,ax=None,h=None,minalpha=.25):
  batch_size = predcpu['stateprob'].shape[0]
  contextl = predcpu['stateprob'].shape[1]
  nstates = predcpu['stateprob'].shape[2]
  if ax is None:
    fig,ax = plt.subplots(1,1)
    
    
  Xkp_true = train_dataset.get_Xkp(example['input'][0,...].numpy(),
                                   example['init'].numpy(),
                                   example['labels'][:tplot+1,...].numpy(),
                                   example['scale'].numpy())
  Xkp_true = Xkp_true[...,0]
  
  order = torch.argsort(predcpu['stateprob'][0,tplot,:])
  rank = torch.argsort(order)
  labels = example['labels'][:tplot,:]
  state_cmap = lambda x: plt.get_cmap("tab10")(rank[x]%10)
  
  if h is None:
    h = {'kpt_true': None, 'kpt_state': [None,]*nstates, 
         'edge_true': None, 'edge_state': [None,]*nstates}
  h['kpt_true'],h['edge_true'],_,_,_ = mabe.plot_fly(Xkp_true[:,:,-1],
                                                     skel_lw=2,color=[0,0,0],
                                                     ax=ax,hkpt=h['kpt_true'],hedge=h['edge_true'])
  for i in range(nstates):

    labelspred = torch.cat((labels,predcpu['perstate'][0,[tplot,],:,i]),dim=0)
  
    Xkp_pred = train_dataset.get_Xkp(example['input'][0,...].numpy(),
                                     example['init'].numpy(),
                                     labelspred,
                                     example['scale'].numpy())
    Xkp_pred = Xkp_pred[...,0]
    p = predcpu['stateprob'][0,tplot,i].item()
    alpha = minalpha + p*(1-minalpha)
    color = state_cmap(i)
    h['kpt_state'][i],h['edge_state'][i],_,_,_ = mabe.plot_fly(Xkp_pred[:,:,-1],
                                                               skel_lw=2,color=color,
                                                               ax=ax,hkpt=h['kpt_state'][i],
                                                               hedge=h['edge_state'][i])
    h['edge_state'][i].set_alpha(alpha)
    h['kpt_state'][i].set_alpha(alpha)
    
  return h,ax,fig

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

def debug_plot_sample_inputs(example,simplify_in):

  fig,ax = plt.subplots(3,2)
  inputidxstart = [-.5,]
  inputidxtype = ['movement']
  inputidxstart.append(inputidxstart[-1]+example['labels'].shape[-1])
  inputidxtype.append('pose')
  if simplify_in != 'no_sensory':
    inputidxstart.append(inputidxstart[-1]+np.count_nonzero(featrelative))
    inputidxtype.append('wall_touch')
    inputidxstart.append(inputidxstart[-1]+nkpwall)
    inputidxtype.append('otherflies_vision')
    
  for iplot in range(3):
    ax[iplot,0].cla()
    ax[iplot,1].cla()
    ax[iplot,0].imshow(example['input'][iplot,...],vmin=-3,vmax=3,cmap='coolwarm')
    ax[iplot,0].axis('auto')
    ax[iplot,0].set_title(f'Input {iplot}')
    #ax[iplot,0].set_xticks(inputidxstart)
    for j in range(len(inputidxtype)):
      ax[iplot,0].plot([inputidxstart[j],]*2,[-.5,example['input'].shape[1]-.5],'k-')
      ax[iplot,0].text(inputidxstart[j],example['input'].shape[1]-1,inputidxtype[j],horizontalalignment='left')
    #ax[iplot,0].set_xticklabels(inputidxtype)
    ax[iplot,1].imshow(example['labels'][iplot,...],vmin=-3,vmax=3,cmap='coolwarm')
    ax[iplot,1].axis('auto')
    ax[iplot,1].set_title(f'Labels {iplot}')
  return fig,ax
  
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
  
def debug_plot_predictions_vs_labels(predv,labelsv,outnames,maskidx=None,ax=None,ylim_nstd=3):
  
  ismasked = maskidx is not None and len(maskidx) > 0
  d_output = predv.shape[-1]
  
  if ax is None:
    fig,ax = plt.subplots(d_output,1,sharex='all',sharey='row')
    fig.set_figheight(20)
    fig.set_figwidth(20)
    plt.tight_layout(h_pad=0)

  pred_cmap = lambda x: plt.get_cmap("tab10")(x%10)
  for i in range(d_output):
    ax[i].cla()
    ax[i].plot(labelsv[:,i],'k-',label='True')
    if ismasked:
      ax[i].plot(maskidx,predv[maskidx,i],'.',color=pred_cmap(i),label='Pred')
    else:
      ax[i].plot(predv[:,i],'-',color=pred_cmap(i),label='Pred')
    ax[i].set_ylim([-ylim_nstd,ylim_nstd])
    ax[i].set_xlim([0,labelsv.shape[0]])
    ti = ax[i].set_title(outnames[i],y=1.0,pad=-14,color=pred_cmap(i),loc='left')
    plt.setp(ti,color=pred_cmap(i))
    
  return fig,ax



def animate_pose(Xkps,focusflies=[],ax=None,fig=None,t0=0,
                 figsizebase=11,ms=6,lw=1,focus_ms=12,focus_lw=3,
                 titletexts={},savevidfile=None,fps=30):

  nax = len(Xkps)
  naxc = int(np.ceil(np.sqrt(nax)))
  naxr = int(np.ceil(nax/naxc))
  
  # get rid of blank flies
  Xkp = list(Xkps.values())[0]
  T = Xkp.shape[-2]
  isreal = mabe.get_real_flies(Xkp)
  nflies = Xkp.shape[-1]
  for Xkp in Xkps.values():
    assert(nflies == Xkp.shape[-1])
    isreal = isreal | mabe.get_real_flies(Xkp)

  for k,v in Xkps.items():
    Xkps[k] = v[...,isreal]
  nflies = np.count_nonzero(isreal)

  minv = -mabe.ARENA_RADIUS_MM*1.01
  maxv = mabe.ARENA_RADIUS_MM*1.01
  
  h = {}

  trel = 0
  t = t0+trel
  if ax is None:
    if fig is None:
      fig,ax = plt.subplots(naxr,naxc)
      fig.set_figheight(figsizebase*naxr)
      fig.set_figwidth(figsizebase*naxc)
    else:
      ax = fig.subplots(naxr,naxc)
  else:
    assert(ax.size>=nax)
  ax = ax.flatten()

  h['kpt'] = []
  h['edge'] = []
  h['ti'] = []
  
  titletext_ts = np.array(list(titletexts.keys()))
  
  if trel in titletexts:
    titletext_str = titletexts[trel]
  else:
    titletext_str = ''

  for i,k in enumerate(Xkps):
    hkpt,hedge,_,_,_ = mabe.plot_flies(Xkps[k][...,trel,:],ax=ax[i],kpt_ms=ms,skel_lw=lw)
    for j in focusflies:
      hkpt[j].set_markersize(focus_ms)
      hedge[j].set_linewidth(focus_lw)
    h['kpt'].append(hkpt)
    h['edge'].append(hedge)
    ax[i].set_aspect('equal')
    mabe.plot_arena(ax=ax[i])
    if i == 0:
      hti = ax[i].set_title(f'{titletext_str} {k}, t = {t}')
    else:
      hti = ax[i].set_title(k)
    h['ti'].append(hti)

    ax[i].set_xlim(minv,maxv)
    ax[i].set_ylim(minv,maxv)

  fig.tight_layout()

  hlist = []
  for hcurr in h.values():
    if type(hcurr) == list:
      hlist+=hcurr
    else:
      hlist+=[hcurr,]

  def update(trel):

    t = t0+trel
    if np.any(titletext_ts<=trel):
      titletext_t = np.max(titletext_ts[titletext_ts<=trel])
      titletext_str = titletexts[titletext_t]
    else:
      titletext_str = ''

    for i,k in enumerate(Xkps):
      mabe.plot_flies(Xkps[k][...,trel,:],ax=ax[0],hkpts=h['kpt'][i],hedges=h['edge'][i])
      if i == 0:
        h['ti'][i].set_text(f'{titletext_str} {k}, t = {t}')
      else:
        h['ti'][i].set_text(k)
    
    return hlist

  ani = animation.FuncAnimation(fig, update, frames=range(T))

  if savevidfile is not None:
    print('Saving animation to file %s...'%savevidfile)
    writer = animation.PillowWriter(fps=30)
    ani.save(savevidfile,writer=writer)
    print('Finished writing.')

  return ani

  
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
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_202302060458.pth')
# CLM, trained with dropout = 0.8 on movement
#loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flyclm_71G01_male_epoch100_20230228T193725.pth'
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
#MODELSTATETYPE = 'prob'
#MODELSTATETYPE = 'best'
MODELSTATETYPE = None
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
if MODELSTATETYPE == 'prob':
  MODEL_ARGS['nstates'] = 32
  MODEL_ARGS['minstateprob'] = 1/MODEL_ARGS['nstates']
elif MODELSTATETYPE == 'best':
  MODEL_ARGS['nstates'] = 8
  
LAST_PTEACHER_FORCE = .001
GAMMA_TEACHER_FORCE = LAST_PTEACHER_FORCE**(1./(NUM_TRAIN_EPOCHS-1))
  
# whether to try to simplify the task, and how
SIMPLIFY_OUT = None #'global'
SIMPLIFY_IN = None
#SIMPLIFY_IN = 'no_sensory'

NPLOT = 32*512*100 // (BATCH_SIZE*CONTEXTL)
SAVE_EPOCH = 5

PDROPOUT_PAST = .8

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
reparamfun = lambda x,id,flynum: compute_features(x,id,flynum,scale_perfly,outtype=np.float32,
                                                  simplify_out=SIMPLIFY_OUT,simplify_in=SIMPLIFY_IN)

val_reparamfun = lambda x,id,flynum,**kwargs: compute_features(x,id,flynum,val_scale_perfly,
                                                               outtype=np.float32,
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

train_dataset = FlyMLMDataset(X,max_mask_length=MAX_MASK_LENGTH,pmask=PMASK,masktype=MASKTYPE,simplify_out=SIMPLIFY_OUT,
                              pdropout_past=PDROPOUT_PAST)
train_dataset.zscore()

mu_input = train_dataset.mu_input.copy()
sig_input = train_dataset.sig_input.copy()
mu_labels = train_dataset.mu_labels.copy()
sig_labels = train_dataset.sig_labels.copy()

val_dataset = FlyMLMDataset(valX,max_mask_length=MAX_MASK_LENGTH,pmask=PMASK,
                            masktype=MASKTYPE,simplify_out=SIMPLIFY_OUT,maskflag=PDROPOUT_PAST>0)
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

example = next(iter(train_dataloader))

# debug plots

# plot to visualize input features
fig,ax = debug_plot_sample_inputs(example,SIMPLIFY_IN)

# plot to check that we can get poses from examples
h,ax,fig = debug_plot_batch_pose(example,train_dataset,data=data)
ax[-1,0].set_xlabel('Train')

# plot to visualize motion outputs
axtraj,figtraj = debug_plot_batch_traj(example,train_dataset,data=data,
                                       label_true='Label',
                                       label_pred='Raw')
axtraj[0].set_title('Train')

# debug plots for validation data set
example = next(iter(val_dataloader))
valh,valax,valfig = debug_plot_batch_pose(example,val_dataset,data=valdata)
valax[-1,0].set_xlabel('Val')
valaxtraj,valfigtraj = debug_plot_batch_traj(example,val_dataset,data=valdata,
                                       label_true='Label',
                                       label_pred='Raw')
valaxtraj[0].set_title('Val')

hstate = None
axstate = None
figstate = None
valhstate = None
valaxstate = None
valfigstate = None

plt.show()
plt.pause(.001)

if MODELSTATETYPE == 'prob':
  model = TransformerStateModel(d_feat,d_output,**MODEL_ARGS).to(device)
  criterion = prob_causal_criterion
elif MODELSTATETYPE == 'min':
  model = TransformerBestStateModel(d_feat,d_output,**MODEL_ARGS).to(device)  
  criterion = min_causal_criterion
else:
  model = TransformerModel(d_feat,d_output,**MODEL_ARGS).to(device)
  if MODELTYPE == 'mlm':
    criterion = masked_criterion
  else:
    criterion = causal_criterion

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

if MODELSTATETYPE is not None:
  modeltype_str = f'{MODELSTATETYPE}_{MODELTYPE}'
else:
  modeltype_str = MODELTYPE

if loadmodelfile is None:
  
  lossfig = plt.figure()
  lossax = plt.gca()
  htrainloss, = lossax.plot(train_loss_epoch,'.-',label='Train')
  hvalloss, = lossax.plot(val_loss_epoch,'.-',label='Val')
  lossax.set_xlabel('Epoch')
  lossax.set_ylabel('Loss')
  lossax.legend()
  
  savetime = datetime.datetime.now()
  savetime = savetime.strftime('%Y%m%dT%H%M%S')
  
  pteacherforce = 1.

  for epoch in range(NUM_TRAIN_EPOCHS):
    
    model.train()
    tr_loss = torch.tensor(0.0).to(device)

    nmask_train = 0
    for step, example in enumerate(train_dataloader):
      
      if pteacherforce < 1:
        pass
      
      pred = model(example['input'].to(device=device),train_src_mask)
      if MODELTYPE == 'mlm':
        loss = criterion(example['labels'].to(device=device),pred,
                         example['mask'].to(device))
      else:
        loss = criterion(example['labels'].to(device=device),pred)
        
      loss.backward()
      if MODELTYPE == 'mlm':
        nmask_train += torch.count_nonzero(example['mask'])
      else:
        nmask_train += BATCH_SIZE*contextl

      if step % NPLOT == 0:
        if MODELSTATETYPE == 'prob':
          pred1 = model.maxpred({k: v.detach() for k,v in pred.items()})
        elif MODELSTATETYPE == 'best':
          pred1 = model.randpred(pred.detach())
        else:
          pred1 = pred.detach()
        debug_plot_batch_pose(example,train_dataset,pred=pred1.cpu(),h=h,ax=ax,fig=fig)
        debug_plot_batch_traj(example,train_dataset,pred=pred1.cpu(),ax=axtraj,fig=figtraj)
        if MODELSTATETYPE == 'prob':
          hstate,axstate,figstate = debug_plot_batch_state(pred['stateprob'].detach().cpu(),nsamplesplot=3,
                                                            h=hstate,ax=axstate,fig=figstate)
          axstate[0].set_title('Train')

        axtraj[0].set_title('Train')
        valexample = next(iter(val_dataloader))
        with torch.no_grad():
          valpred = model(valexample['input'].to(device=device),train_src_mask)
          if MODELSTATETYPE == 'prob':
            valpred1 = model.maxpred(valpred)
          elif MODELSTATETYPE == 'best':
            valpred1 = model.randpred(valpred)
          else:
            valpred1 = valpred
          debug_plot_batch_pose(valexample,val_dataset,pred=valpred1.cpu(),h=valh,ax=valax,fig=valfig)
          debug_plot_batch_traj(valexample,val_dataset,pred=valpred1.cpu(),ax=valaxtraj,fig=valfigtraj)
          valaxtraj[0].set_title('Val')
          if MODELSTATETYPE == 'prob':
            valhstate,valaxstate,valfigstate = debug_plot_batch_state(valpred['stateprob'].cpu(),nsamplesplot=3,
                                                                      h=valhstate,ax=valaxstate,fig=valfigstate)
            valaxstate[0].set_title('Val')

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
          loss = criterion(example['labels'].to(device=device),pred,
                           example['mask'].to(device))
          nmask_val += torch.count_nonzero(example['mask'])
        elif MODELTYPE == 'clm':
          loss = criterion(example['labels'].to(device=device),pred)
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

    train_dataset = FlyMLMDataset(X,max_mask_length=MAX_MASK_LENGTH,pmask=PMASK,masktype=MASKTYPE,simplify_out=SIMPLIFY_OUT,
                                  pdropout_past=PDROPOUT_PAST)
    train_dataset.zscore(mu_input=mu_input,sig_input=sig_input,mu_labels=mu_labels,sig_labels=sig_labels)
    print('New training data set created')

    if (epoch+1)%SAVE_EPOCH == 0:
      savefile = os.path.join(savedir,f'fly{modeltype_str}_{"_".join(categories)}_epoch{epoch+1}_{savetime}.pth')
      print(f'Saving to file {savefile}')
      save_model(savefile,model,lr_optimizer=optimizer,scheduler=lr_scheduler,loss=train_loss_epoch,val_loss=val_loss_epoch)

    pteacherforce *= GAMMA_TEACHER_FORCE

  savefile = os.path.join(savedir,f'fly{modeltype_str}_{"_".join(categories)}_epoch{epoch+1}_{savetime}.pth')
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

# compute predictions and labels for all validation data using default masking
all_pred = []
all_mask = []
all_labels = []
with torch.no_grad():
  val_loss = torch.tensor(0.0).to(device)
  for example in val_dataloader:
    pred = model(example['input'].to(device=device),train_src_mask)
    if MODELSTATETYPE == 'prob':
      pred = model.maxpred(pred)
    elif MODELSTATETYPE == 'best':
      pred = model.randpred(pred)
    all_pred.append(pred.cpu())
    if 'mask' in example:
      all_mask.append(example['mask'])
    all_labels.append(example['labels'])

# plot comparison between predictions and labels on validation data
nplot = min(len(all_labels),8000//BATCH_SIZE//CONTEXTL+1)

predv,labelsv,maskv = stackhelper(all_pred,all_labels,all_mask,nplot)
if maskv is not None and len(maskv) > 0:
  maskidx = torch.nonzero(maskv)[:,0]
else:
  maskidx = None

outnames = val_dataset.get_outnames()
fig,ax = debug_plot_predictions_vs_labels(predv,labelsv,outnames,maskidx)

DEBUG = False
burnin = CONTEXTL-1
contextlpad = burnin + 1
# all frames for the main fly must have real data
allisdata = interval_all(valdata['isdata'],contextlpad)
isnotsplit = interval_all(valdata['isstart']==False,contextlpad-1)[1:,...]
canstart = np.logical_and(allisdata,isnotsplit)
#flynum = 0
#t0 = np.nonzero(canstart[:,flynum])[0][360-CONTEXTL]
TPRED = 1000
flynum = 2
t0 = np.nonzero(canstart[:,flynum])[0][1500-CONTEXTL]
fliespred = np.array([flynum,])

Xkp_true = valdata['X'][...,t0:t0+TPRED,:].copy()
Xkp = Xkp_true.copy()

#fliespred = np.nonzero(mabe.get_real_flies(Xkp))[0]
ids = valdata['ids'][t0,fliespred]
scales = val_scale_perfly[:,ids]

model.eval()

Xkp_pred = predict_open_loop(Xkp,fliespred,scales,burnin,val_dataset,model,maxcontextl=np.inf,debug=DEBUG)

Xkps = {'Pred': Xkp_pred.copy(),'True': Xkp_true.copy()}
focusflies = fliespred
titletexts = {0: 'Initialize', burnin: ''}

vidtime = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
savevidfile = os.path.join(savedir,f'samplevideo_{"_".join(categories)}_{vidtime}.gif')

ani = animate_pose(Xkps,focusflies=focusflies,t0=t0,titletexts=titletexts)

