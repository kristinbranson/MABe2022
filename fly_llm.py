import math
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm,animation,colors
import copy
import typing
import tqdm
from itertools import compress
import re
import MABeFlyUtils as mabe
import torch
import transformers
import warnings
import datetime
import collections
# import sklearn.preprocessing
import sklearn.cluster
import sklearn.decomposition
import json
import argparse
import pathlib
import pickle
import gzip

matplotlib.use('tkagg')
plt.ion()

print('fly_llm...')

codedir = pathlib.Path(__file__).parent.resolve()
DEFAULTCONFIGFILE = os.path.join(codedir,'config_fly_llm_default.json')
assert os.path.exists(DEFAULTCONFIGFILE)

legtipnames = [
  'right_front_leg_tip',
  'right_middle_leg_tip',
  'right_back_leg_tip',
  'left_back_leg_tip',
  'left_middle_leg_tip',
  'left_front_leg_tip',
]

vision_kpnames_v1 = [
  'antennae_midpoint',
  'tip_abdomen',
  'left_middle_femur_base',
  'right_middle_femur_base',
]

touch_other_kpnames_v1 = [
  'antennae_midpoint',
  'left_front_thorax',
  'right_front_thorax',
  'base_thorax',
  'tip_abdomen',
]

SENSORY_PARAMS = {
  'n_oma': 72,
  'inner_arena_radius': 17.5, # in mm
  'outer_arena_radius': mabe.ARENA_RADIUS_MM,
  'arena_height': 3.5,
  'otherflies_vision_exp': .6,
  'touch_kpnames': mabe.keypointnames,
  #'touch_kpnames': legtipnames,
  'vision_kpnames': vision_kpnames_v1,
  'touch_other_kpnames': touch_other_kpnames_v1,
  'compute_otherflies_touch': True,
  'otherflies_touch_exp': 1.3,
  'otherflies_touch_mult': 0.3110326159171111, # set 20230807 based on courtship male data
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
      print(f'loading {key}')
      data[key] = data1[key]
  print('data loaded')
    
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
def chunk_data(data,contextl,reparamfun,npad=1):
  
  contextlpad = contextl + npad
  
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
  nframestotal = 0
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
      xcurr = reparamfun(data['X'][...,t0:t1+1,:],id,flynum,npad=npad)
      xcurr['metadata'] = {'flynum': flynum, 'id': id, 't0': t0, 'videoidx': data['videoidx'][t0,0], 'frame0': data['frames'][t0,0]}
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
      nframestotal += contextl

  print(f'In total {nframestotal} frames of data after chunking')

  return X

def compute_noise_params(data,scale_perfly,sig_tracking=.25/mabe.PXPERMM,
                         simplify_out=None,compute_pose_vel=True):

  # contextlpad = 2
  
  # # all frames for the main fly must have real data
  # allisdata = interval_all(data['isdata'],contextlpad)
  # isnotsplit = interval_all(data['isstart']==False,contextlpad-1)[1:,...]
  # canstart = np.logical_and(allisdata,isnotsplit)

  # X is nkeypts x 2 x T x nflies
  nkeypoints = data['X'].shape[0]
  T = data['X'].shape[2]
  maxnflies = data['X'].shape[3]

  alld = 0.
  n = 0
  # loop through ids
  print('Computing noise parameters...')
  for flynum in tqdm.trange(maxnflies):
    idx0 = data['isdata'][:,flynum] & (data['isstart'][:,flynum]==False)
    # bout starts and ends
    t0s = np.nonzero(np.r_[idx0[0],(idx0[:-1]==False) & (idx0[1:]==True)])[0]
    t1s = np.nonzero(np.r_[(idx0[:-1]==True) & (idx0[1:]==False),idx0[-1]])[0]
    
    for i in range(len(t0s)):
      t0 = t0s[i]
      t1 = t1s[i]
      id = data['ids'][t0,flynum]
      scale = scale_perfly[:,id]
      xkp = data['X'][:,:,t0:t1+1,flynum]
      relpose,globalpos = compute_pose_features(xkp,scale)
      movement = compute_movement(relpose=relpose,globalpos=globalpos,simplify=simplify_out,compute_pose_vel=compute_pose_vel)
      nu = np.random.normal(scale=sig_tracking,size=xkp.shape)
      relpose_pert,globalpos_pert = compute_pose_features(xkp+nu,scale)
      movement_pert = compute_movement(relpose=relpose_pert,globalpos=globalpos_pert,simplify=simplify_out,compute_pose_vel=compute_pose_vel)
      alld += np.nansum((movement_pert-movement)**2.,axis=1)
      ncurr = np.sum((np.isnan(movement)==False),axis=1)
      n+=ncurr
  
  epsilon = np.sqrt(alld/n)
  
  return epsilon.flatten()

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


def plot_scale_stuff(data,scale_perfly):
  
  eps_sex = .05
  nbins = 20
  axlim_prctile = .5
  catidx = data['categories'].index('female')

  maxid = np.max(data['ids'])
  maxnflies = data['X'].shape[3]
  fracfemale = np.zeros(maxid+1)
  nframes = np.zeros(maxid+1)
  minnframes = 40000
  prctiles_compute = np.array([50,75,90,95,99,99.5,99.9])
  midleglength = np.zeros((maxid+1,len(prctiles_compute)))

  for flynum in range(maxnflies):

    idscurr = np.unique(data['ids'][data['ids'][:,flynum]>=0,flynum])
    for id in idscurr:
      idx = data['ids'][:,flynum] == id
      fracfemale[id] = np.count_nonzero(data['y'][catidx,idx,flynum]==1) / np.count_nonzero(idx)
      nframes[id] = np.count_nonzero(idx)
      xcurr = data['X'][:,:,idx,flynum]
      midtip = xcurr[mabe.keypointnames.index('left_middle_leg_tip'),:]
      midbase = xcurr[mabe.keypointnames.index('left_middle_femur_base'),:]
      lmidl = np.sqrt(np.sum((midtip-midbase)**2,axis=0))
      midtip = xcurr[mabe.keypointnames.index('right_middle_leg_tip'),:]
      midbase = xcurr[mabe.keypointnames.index('right_middle_femur_base'),:]
      rmidl = np.sqrt(np.sum((midtip-midbase)**2,axis=0))
      midleglength[id,:] = np.percentile(np.hstack((lmidl,rmidl)),prctiles_compute)

  plotnames = ['thorax_width', 'thorax_length', 'abdomen_length', 'head_width', 'head_height']
  plotidx = np.array([v in plotnames for v in mabe.scalenames])
  #plotidx = np.array([re.search('std',s) is None for s in mabe.scalenames])
  plotidx = np.nonzero(plotidx)[0]
  plotfly = nframes >= minnframes
  fig,ax = plt.subplots(len(plotidx),len(plotidx))
  fig.set_figheight(20)
  fig.set_figwidth(20)
  
  idxfemale = plotfly & (fracfemale>=1-eps_sex)
  idxmale = plotfly & (fracfemale<=eps_sex)

  lims = np.percentile(scale_perfly[:,plotfly],[axlim_prctile,100-axlim_prctile],axis=1)
  
  for ii in range(len(plotidx)):
    i = plotidx[ii]
    for jj in range(len(plotidx)):
      j = plotidx[jj]
      if i == j:
        binedges = np.linspace(lims[0,i],lims[1,i],nbins+1)
        ax[ii,ii].hist([scale_perfly[i,idxfemale],scale_perfly[i,idxmale]],
                     bins=nbins,range=(lims[0,i],lims[1,i]),
                     label=['female','male'])
        ax[ii,ii].set_ylabel('N. flies')
      else:
        ax[jj,ii].plot(scale_perfly[i,idxfemale],
                       scale_perfly[j,idxfemale],'.',label='female')
        ax[jj,ii].plot(scale_perfly[i,idxmale],
                       scale_perfly[j,idxmale],'.',label='male')
        ax[jj,ii].set_ylabel(mabe.scalenames[j])
        ax[jj,ii].set_xlabel(mabe.scalenames[i])
        ax[jj,ii].set_ylim(lims[:,j])
      ax[jj,ii].set_xlim(lims[:,i])
      ax[jj,ii].set_xlabel(mabe.scalenames[i])
  ax[0,0].legend()
  ax[0,1].legend()
  fig.tight_layout()
  
  scalefeat = 'thorax_length'
  scalei = mabe.scalenames.index(scalefeat)
  fig,ax = plt.subplots(2,len(prctiles_compute),sharex='row',sharey='row')
  fig.set_figwidth(20)
  fig.set_figheight(8)
  lims = np.percentile(midleglength[plotfly,:].flatten(),[axlim_prctile,100-axlim_prctile])
  for i in range(len(prctiles_compute)):
    ax[0,i].plot(scale_perfly[scalei,idxfemale],midleglength[idxfemale,i],'.',label='female')
    ax[0,i].plot(scale_perfly[scalei,idxmale],midleglength[idxmale,i],'.',label='male')
    ax[0,i].set_xlabel(scalefeat)
    ax[0,i].set_ylabel(f'{prctiles_compute[i]}th %ile middle leg length')
    ax[1,i].hist([midleglength[idxfemale,i],midleglength[idxmale,i]],
                 bins=nbins,range=(lims[0],lims[1]),label=['female','male'],
                 density=True)
    ax[1,i].set_xlabel(f'{prctiles_compute[i]}th %ile middle leg length')
    ax[1,i].set_ylabel('Density')
  ax[0,0].legend()
  ax[1,0].legend()
  fig.tight_layout()
  
  
  return
  


def pred_apply_fun(pred,fun):
  if isinstance(pred,dict):
    return {k: fun(v) for k,v in pred.items()}
  else:
    return fun(pred)

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
  for pti in range(nkptouch):
    hpts.append(ax.plot(xwall[pti,t],ywall[pti,t],'o')[0])
  ax.set_aspect('equal')
  ax = plt.subplot(1,2,2)
  ax.plot(distleg.flatten(),wall_touch.flatten(),'k.')
  ax.plot([0,SENSORY_PARAMS['inner_arena_radius'],SENSORY_PARAMS['outer_arena_radius']],
          [SENSORY_PARAMS['arena_height'],SENSORY_PARAMS['arena_height'],0],'-')
  for pti in range(nkptouch):
    ax.plot(distleg[pti,t],wall_touch[pti,t],'o',color=hpts[pti].get_color())
  ax.set_aspect('equal')

# compute sensory input
#
# inputs:
# xeye_main: x-coordinate of main fly's position for vision. shape = (T).
# yeye_main: y-coordinate of main fly's position for vision. shape = (T).
# theta_main: orientation of main fly. shape = (T).
# xtouch_main: x-coordinates of main fly's keypoints for computing touch inputs (both wall and other fly). shape = (npts_touch,T).
# ytouch_main: y-coordinates of main fly's keypoints for computing touch inputs (both wall and other fly). shape = (npts_touch,T).
# xvision_other: x-coordinate of keypoints on the other flies for computing vision input. shape = (npts_vision,T,nflies)
# yvision_other: y-coordinate of keypoints on the other flies for computing vision input. shape = (npts_vision,T,nflies)
# xtouch_other: x-coordinate of keypoints on the other flies for computing touch input. shape = (npts_touch_other,T,nflies)
# ytouch_other: y-coordinate of keypoints on the other flies for computing touch input. shape = (npts_touch_other,T,nflies)
#
# outputs:
# otherflies_vision: appearance of other flies to input fly. this is computed as a 
# 1. - np.minimum(1.,SENSORY_PARAMS['otherflies_vision_mult'] * dist**SENSORY_PARAMS['otherflies_vision_exp'])
# where dist is the minimum distance to some point on some other fly x,y_vision_other in each of n_oma directions. 
# shape = (SENSORY_PARAMS['n_oma'],T).  
# wall_touch: height of arena chamber at each keypoint in x,y_touch_main. this is computed as
# np.minimum(SENSORY_PARAMS['arena_height'],np.maximum(0.,SENSORY_PARAMS['arena_height'] - 
# (distleg-SENSORY_PARAMS['inner_arena_radius'])*SENSORY_PARAMS['arena_height']/(SENSORY_PARAMS['outer_arena_radius']-
# SENSORY_PARAMS['inner_arena_radius'])))
# shape = (npts_touch,T), 
# otherflies_touch: information about touch from other flies to input fly. this is computed as
# 1. - np.minimum(1.,SENSORY_PARAMS['otherflies_touch_mult'] * dist**SENSORY_PARAMS['otherflies_touch_exp'])
# where dist is the minimum distance over all other flies from each keypoint in x,y_touch_main to each keypoint in x,y_touch_other
# there are two main difference between this and otherflies_vision. first is this uses multiple keypoints on the main and other flies
# and has an output for each of them. conversely, otherflies_vision has an output for each direction. the second difference is
# based on the parameters in SENSORY_PARAMS. The parameters for touch should be set so that the maximum distance over which there is a 
# signal is about how far any keypoint can be from any of the keypoints in x,y_touch_other, which the maximum distance for 
# vision is over the entire arena. 
# shape = (npts_touch*npts_touch_other,T).

def compute_sensory(xeye_main,yeye_main,theta_main,
                    xtouch_main,ytouch_main,
                    xvision_other,yvision_other,
                    xtouch_other,ytouch_other):

  # increase dimensions if only one frame input
  if xvision_other.ndim < 3:
    T = 1
  else:
    T = xvision_other.shape[2]

  npts_touch = xtouch_main.shape[0]
  npts_vision = xvision_other.shape[0]
  npts_touch_other = xtouch_other.shape[0]
  nflies = xvision_other.shape[1]
  
  xvision_other = np.reshape(xvision_other,(npts_vision,nflies,T))
  yvision_other = np.reshape(yvision_other,(npts_vision,nflies,T))
  xtouch_other = np.reshape(xtouch_other,(npts_touch_other,nflies,T))
  ytouch_other = np.reshape(ytouch_other,(npts_touch_other,nflies,T))
  xeye_main = np.reshape(xeye_main,(1,1,T))
  yeye_main = np.reshape(yeye_main,(1,1,T))
  theta_main = np.reshape(theta_main,(1,1,T))
  xtouch_main = np.reshape(xtouch_main,(npts_touch,T))
  ytouch_main = np.reshape(ytouch_main,(npts_touch,T))
  
  # don't deal with missing data :)    
  assert(np.any(np.isnan(xeye_main))==False)
  assert(np.any(np.isnan(yeye_main))==False)
  assert(np.any(np.isnan(theta_main))==False)
  
  # vision bin size
  step = 2.*np.pi/SENSORY_PARAMS['n_oma']

  # compute other flies view

  # convert to this fly's coord system
  dx = xvision_other-xeye_main
  dy = yvision_other-yeye_main
  
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
  distarena = np.sqrt( xtouch_main**2. + ytouch_main**2 )

  # height of chamber 
  wall_touch = np.zeros(distarena.shape)
  wall_touch[:] = SENSORY_PARAMS['arena_height']
  wall_touch = np.minimum(SENSORY_PARAMS['arena_height'],np.maximum(0.,SENSORY_PARAMS['arena_height'] - (distarena-SENSORY_PARAMS['inner_arena_radius'])*SENSORY_PARAMS['arena_height']/(SENSORY_PARAMS['outer_arena_radius']-SENSORY_PARAMS['inner_arena_radius'])))
  wall_touch[distarena >= SENSORY_PARAMS['outer_arena_radius']] = 0.
  
  # t = 0
  # debug_plot_wall_touch(t,xlegtip_main,ylegtip_main,distleg,wall_touch,params)

  # xtouch_main: npts_touch x T, xtouch_other: npts_touch_other x nflies x T
  if SENSORY_PARAMS['compute_otherflies_touch']:
    dx = xtouch_main.reshape((npts_touch,1,1,T)) - xtouch_other.reshape((1,npts_touch_other,nflies,T)) 
    dy = ytouch_main.reshape((npts_touch,1,1,T)) - ytouch_other.reshape((1,npts_touch_other,nflies,T)) 
    d = np.sqrt(np.nanmin(dx**2 + dy**2,axis=2)).reshape(npts_touch*npts_touch_other,T)
    otherflies_touch = 1. - np.minimum(1.,SENSORY_PARAMS['otherflies_touch_mult'] * d**SENSORY_PARAMS['otherflies_touch_exp'])
  else:
    otherflies_touch = None
  
  return (otherflies_vision, wall_touch, otherflies_touch)

def compute_sensory_torch(xeye_main,yeye_main,theta_main,
                          xtouch_main,ytouch_main,
                          xvision_other,yvision_other,
                          xtouch_other,ytouch_other):

  """
  compute sensory input
  compute_sensory_torch(xeye_main,yeye_main,theta_main,
                          xtouch_main,ytouch_main,
                          xvision_other,yvision_other,
                          xtouch_other,ytouch_other)

  inputs:
  xeye_main: x-coordinate of main fly's position for vision. shape = (T).
  yeye_main: y-coordinate of main fly's position for vision. shape = (T).
  theta_main: orientation of main fly. shape = (T).
  xtouch_main: x-coordinates of main fly's keypoints for computing touch inputs (both wall and other fly). shape = (npts_touch,T).
  ytouch_main: y-coordinates of main fly's keypoints for computing touch inputs (both wall and other fly). shape = (npts_touch,T).
  xvision_other: x-coordinate of keypoints on the other flies for computing vision input. shape = (npts_vision,T,nflies)
  yvision_other: y-coordinate of keypoints on the other flies for computing vision input. shape = (npts_vision,T,nflies)
  xtouch_other: x-coordinate of keypoints on the other flies for computing touch input. shape = (npts_touch_other,T,nflies)
  ytouch_other: y-coordinate of keypoints on the other flies for computing touch input. shape = (npts_touch_other,T,nflies)

  outputs:
  otherflies_vision: appearance of other flies to input fly. this is computed as a 
  1. - np.minimum(1.,SENSORY_PARAMS['otherflies_vision_mult'] * dist**SENSORY_PARAMS['otherflies_vision_exp'])
  where dist is the minimum distance to some point on some other fly x,y_vision_other in each of n_oma directions. 
  shape = (SENSORY_PARAMS['n_oma'],T).  
  wall_touch: height of arena chamber at each keypoint in x,y_touch_main. this is computed as
  np.minimum(SENSORY_PARAMS['arena_height'],np.maximum(0.,SENSORY_PARAMS['arena_height'] - 
  (distleg-SENSORY_PARAMS['inner_arena_radius'])*SENSORY_PARAMS['arena_height']/(SENSORY_PARAMS['outer_arena_radius']-
  SENSORY_PARAMS['inner_arena_radius'])))
  shape = (npts_touch,T), 
  otherflies_touch: information about touch from other flies to input fly. this is computed as
  1. - np.minimum(1.,SENSORY_PARAMS['otherflies_touch_mult'] * dist**SENSORY_PARAMS['otherflies_touch_exp'])
  where dist is the minimum distance over all other flies from each keypoint in x,y_touch_main to each keypoint in x,y_touch_other
  there are two main difference between this and otherflies_vision. first is this uses multiple keypoints on the main and other flies
  and has an output for each of them. conversely, otherflies_vision has an output for each direction. the second difference is
  based on the parameters in SENSORY_PARAMS. The parameters for touch should be set so that the maximum distance over which there is a 
  signal is about how far any keypoint can be from any of the keypoints in x,y_touch_other, which the maximum distance for 
  vision is over the entire arena. 
  shape = (npts_touch*npts_touch_other,T).
  """

  device = xeye_main.device
  dtype = xeye_main.dtype

  # increase dimensions if only one frame input
  if xvision_other.ndim < 3:
    T = 1
  else:
    T = xvision_other.shape[2]

  npts_touch = xtouch_main.shape[0]
  npts_vision = xvision_other.shape[0]
  npts_touch_other = xtouch_other.shape[0]
  nflies = xvision_other.shape[1]

  xvision_other = torch.reshape(xvision_other,(npts_vision,nflies,T))
  yvision_other = torch.reshape(yvision_other,(npts_vision,nflies,T))
  xtouch_other = torch.reshape(xtouch_other,(npts_touch_other,nflies,T))
  ytouch_other = torch.reshape(ytouch_other,(npts_touch_other,nflies,T))
  xeye_main = torch.reshape(xeye_main,(1,1,T))
  yeye_main = torch.reshape(yeye_main,(1,1,T))
  theta_main = torch.reshape(theta_main,(1,1,T))
  xtouch_main = torch.reshape(xtouch_main,(npts_touch,T))
  ytouch_main = torch.reshape(ytouch_main,(npts_touch,T))

  # don't deal with missing data :)
  assert(torch.any(torch.isnan(xeye_main))==False)
  assert(torch.any(torch.isnan(yeye_main))==False)
  assert(torch.any(torch.isnan(theta_main))==False)
  
  # vision bin size
  step = 2.*torch.pi/SENSORY_PARAMS['n_oma']

  # compute other flies view

  # convert to this fly's coord system
  dx = xvision_other-xeye_main
  dy = yvision_other-yeye_main
  
  # distance
  dist = torch.sqrt(dx**2+dy**2)
  
  # angle in the original coordinate system
  angle0 = torch.arctan2(dy,dx)  
  
  # subtract off angle of main fly
  angle = angle0 - theta_main
  angle = torch.fmod(angle + torch.pi, 2 * torch.pi) - torch.pi

  # which other flies pass beyond the -pi to pi border
  isbackpos = angle > torch.pi/2
  isbackneg = angle < -torch.pi/2
  isfront = torch.abs(angle) <= torch.pi/2
  idxmod = torch.any(isbackpos,dim=0) & torch.any(isbackneg,dim=0) & (~torch.any(isfront,dim=0))

  # bin - npts x nflies x T
  b_all = torch.floor((angle+np.pi)/step)
  
  # bin range
  # shape: nflies x T
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    minb = torch.nanmin(b_all, dim=0)
    maxb = torch.nanmax(b_all, dim=0)
    mind = torch.nanmin(dist, dim=0)
    
  # n_oma x 1 x 1
  tmpbins = torch.arange(SENSORY_PARAMS['n_oma'],dtype=dtype,device=device)[:,None,None]

  # n_oma x nflies x T
  mindrep = mind[None,...].repeat((SENSORY_PARAMS['n_oma'],1,1))
  mask = (tmpbins >= minb[None,...]) & (tmpbins <= maxb[None,...])

  if torch.any(idxmod):
    # this is complicated!!
    # find the max bin for negative angles
    # and the min bin for positive angles
    # store them in min and max for consistency with non-wraparound
    isbackpos1 = isbackpos[:,idxmod]
    isbackneg1 = isbackneg[:,idxmod]
    bmodneg = b_all[:,idxmod]
    bmodneg[isbackpos1] = torch.nan
    minbmod = torch.nanmax(bmodneg,dim=0)
    bmodpos = b_all[:,idxmod]
    bmodpos[isbackneg1] = torch.nan
    maxbmod = torch.nanmin(bmodpos,dim=0)
    mask[:,idxmod] = (tmpbins[...,0] >= maxbmod.unsqueeze(0)) | (tmpbins[...,0] <= minbmod.unsqueeze(0))

  otherflies_vision = torch.nanmin(torch.where(mask, mindrep, torch.full_like(mindrep, float('inf'))), dim=1, keepdim=False, initial=float('inf'))
  otherflies_vision = 1. - torch.min(1., SENSORY_PARAMS['otherflies_vision_mult'] * otherflies_vision**SENSORY_PARAMS['otherflies_vision_exp'])

  # t = 249
  # debug_plot_otherflies_vision(t,xother,yother,xeye_main,yeye_main,theta_main,
  #                                 angle0,angle,dist,b_all,otherflies_vision,params)

  # distance from center of arena
  # center of arena is assumed to be [0,0]
  distleg = torch.sqrt( xtouch_main**2. + ytouch_main**2 )

  # height of chamber 
  wall_touch = torch.zeros(distleg.shape,dtype=dtype,device=device)
  wall_touch[:] = SENSORY_PARAMS['arena_height']
  wall_touch = torch.minimum(SENSORY_PARAMS['arena_height'],torch.maximum(0.,SENSORY_PARAMS['arena_height'] - (distleg-SENSORY_PARAMS['inner_arena_radius'])*SENSORY_PARAMS['arena_height']/(SENSORY_PARAMS['outer_arena_radius']-SENSORY_PARAMS['inner_arena_radius'])))
  wall_touch[distleg >= SENSORY_PARAMS['outer_arena_radius']] = 0.
  
  # t = 0
  # debug_plot_wall_touch(t,xlegtip_main,ylegtip_main,distleg,wall_touch,params)

  # xtouch_main: npts_touch x T, xtouch_other: npts_touch_other x nflies x T
  if SENSORY_PARAMS['compute_otherflies_touch']:
    dx = xtouch_main.reshape((npts_touch,1,1,T)) - xtouch_other.reshape((1,npts_touch_other,nflies,T)) 
    dy = ytouch_main.reshape((npts_touch,1,1,T)) - ytouch_other.reshape((1,npts_touch_other,nflies,T)) 
    d = torch.sqrt(torch.nanmin(dx**2 + dy**2,dim=2)).reshape(npts_touch*npts_touch_other,T)
    otherflies_touch = 1. - torch.minimum(1.,SENSORY_PARAMS['otherflies_touch_mult'] * d**SENSORY_PARAMS['otherflies_touch_exp'])
  else:
    otherflies_touch = None
  
  return (otherflies_vision, wall_touch, otherflies_touch)

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

kpvision_other = [mabe.keypointnames.index(x) for x in SENSORY_PARAMS['vision_kpnames']]
kpeye = mabe.keypointnames.index('antennae_midpoint')
kptouch = [mabe.keypointnames.index(x) for x in SENSORY_PARAMS['touch_kpnames']]
nkptouch = len(kptouch)
kptouch_other = [mabe.keypointnames.index(x) for x in SENSORY_PARAMS['touch_other_kpnames']]
nkptouch_other = len(kptouch_other)

narena = 2**10
theta_arena = np.linspace(-np.pi,np.pi,narena+1)[:-1]

def get_sensory_feature_shapes(simplify=None):
  idx = collections.OrderedDict()
  sz = collections.OrderedDict()
  i0 = 0
  i1 = nrelative
  idx['pose'] = [i0,i1]
  sz['pose'] = (nrelative,)
  
  if simplify is None:
    i0 = i1
    i1 = i0+nkptouch
    idx['wall_touch'] = [i0,i1]
    sz['wall_touch'] = (nkptouch,)
    i0 = i1
    i1 = i0 + SENSORY_PARAMS['n_oma']
    idx['otherflies_vision'] = [i0,i1]
    sz['otherflies_vision'] = (SENSORY_PARAMS['n_oma'],)
    i0 = i1
    i1 = i0 + nkptouch*nkptouch_other
    idx['otherflies_touch'] = [i0,i1]
    sz['otherflies_touch'] = (nkptouch*nkptouch_other,)
  return idx,sz

def get_sensory_feature_idx(simplify=None):
  idx,_ = get_sensory_feature_shapes()
  return idx

def split_features(X,simplify=None,axis=-1):
  res = {}
  idx = get_sensory_feature_idx(simplify)
  for k,v in idx.items():
    if torch.is_tensor(X):
      res[k] = torch.index_select(X,axis,torch.tensor(range(v[0],v[1])))
    else:
      res[k] = X.take(range(v[0],v[1]),axis=axis)

  return res

def unpack_input(input,featidx,sz,dim=-1):
  
  res = {}
  idx = [slice(None),]*input.ndim
  sz0 = input.shape
  if dim < 0:
    dim = input.ndim+dim
  for k,v in featidx.items():
    idx[dim] = slice(v[0],v[1])
    newsz = sz0[:dim] + sz[k] + sz0[dim+1:]
    res[k] = input[idx].reshape(newsz)
  
  return res

def combine_relative_global(Xrelative,Xglobal,axis=-1):
  X = np.concatenate((Xglobal,Xrelative),axis=axis)
  return X

def combine_relative_global_pose(relpose,globalpos):
  sz = (nfeatures,)+relpose.shape[1:]
  posefeat = np.zeros(sz,dtype=relpose.dtype)
  posefeat[featrelative,...] = relpose
  posefeat[featglobal,...] = globalpos
  return posefeat

def compute_pose_features(X,scale):
  posefeat = mabe.kp2feat(X,scale)
  relpose = posefeat[featrelative,...]
  globalpos = posefeat[featglobal,...]

  return relpose,globalpos

def boxsum(x,n):
  if n == 1:
    return x
  xtorch = torch.tensor(x[:,None,...])
  y = torch.nn.functional.conv2d(xtorch,torch.ones((1,1,n,1),dtype=xtorch.dtype),padding='valid')
  return y[:,0,...].numpy()

def compute_global_velocity(Xorigin,Xtheta,tspred_global=[1,]):
  """
  compute_global_velocity(Xorigin,Xtheta,tspred_global=[1,])
  compute the movement from t to t+tau for all tau in tspred_global
  Xorigin is the centroid position of the fly, shape = (2,T,nflies)
  Xtheta is the orientation of the fly, shape = (T,nflies)
  returns dXoriginrel,dtheta
  dXoriginrel[i,:,t,fly] is the global position for fly fly at time t+tspred_global[i] in the coordinate system of the fly at time t
  shape = (ntspred_global,2,T,nflies)
  dtheta[i,t,fly] is the total change in orientation for fly fly at time t+tspred_global[i] from time t. this sums per-frame dthetas,
  so it could have a value outside [-pi,pi]. shape = (ntspred_global,T,nflies)
  """
  
  T = Xorigin.shape[1]
  nflies = Xorigin.shape[2]
  ntspred_global = len(tspred_global)

  # global velocity  
  # dXoriginrel[tau,:,t,fly] is the global position for fly fly at time t+tau in the coordinate system of the fly at time t
  dXoriginrel = np.zeros((ntspred_global,2,T,nflies),dtype=Xorigin.dtype)
  dXoriginrel[:] = np.nan
  # dtheta[tau,t,fly] is the change in orientation for fly fly at time t+tau from time t
  dtheta = np.zeros((ntspred_global,T,nflies),dtype=Xtheta.dtype)
  dtheta[:] = np.nan
  # dtheta1[t,fly] is the change in orientation for fly from frame t to t+1
  dtheta1 = mabe.modrange(Xtheta[1:,:]-Xtheta[:-1,:],-np.pi,np.pi)

  for i,toff in enumerate(tspred_global):
    # center and rotate absolute position around position toff frames previous
    dXoriginrel[i,:,:-toff,:] = mabe.rotate_2d_points((Xorigin[:,toff:,:]-Xorigin[:,:-toff,:]).transpose((1,0,2)),Xtheta[:-toff,:]).transpose((1,0,2))
    # compute total change in global orientation in toff frame intervals
    dtheta[i,:-toff,:] = boxsum(dtheta1[None,...],toff)

  return dXoriginrel,dtheta

def compute_relpose_velocity(relpose,tspred_dct=[]):
  """
  compute_relpose_velocity(relpose,tspred_dct=[])
  compute the relative pose movement from t to t+tau for all tau in tspred_dct
  relpose is the relative pose features, shape = (nrelative,T,nflies)
  outputs drelpose, shape = (nrelative,T,ntspred_dct+1,nflies)
  """

  ntspred_dct = len(tspred_dct)
  T = relpose.shape[1]
  nflies = relpose.shape[2]
  
  # drelpose1[:,f,fly] is the change in pose for fly from frame t to t+1
  drelpose1 = relpose[:,1:,:]-relpose[:,:-1,:]
  drelpose1[featangle[featrelative],:,:] = mabe.modrange(drelpose1[featangle[featrelative],:,:],-np.pi,np.pi)
  
  # drelpose[:,tau,t,fly] is the change in pose for fly fly at time t+tau from time t    
  drelpose = np.zeros((nrelative,T,ntspred_dct+1,nflies),dtype=relpose.dtype)
  drelpose[:] = np.nan
  drelpose[:,:-1,0,:] = drelpose1
  
  for i,toff in enumerate(tspred_dct):
    # compute total change in relative pose in toff frame intervals
    drelpose[:,:-toff,i+1,:] = boxsum(drelpose1,toff)
  
  return drelpose

def featidx_to_relfeatidx(featidx):
  return featidx - nglobal

def relfeatidx_to_featidx(relfeatidx):
  return relfeatidx+nglobal

def relfeatidx_to_cossinidx(discreteidx=[]):
  """
  relfeatidx_to_cossinidx(discreteidx=[])
  get the look up table for relative feature index to the cos sin representation index
  discreteidx: list of int, feature indices that are discrete
  returns rel2cossinmap: list, rel2cossinmap[i] is the list of indices for the cos and sin 
  representation of the i-th relative feature
  """
  rel2cossinmap = []
  csi = 0
  for relfeati in range(nrelative):
    feati = relfeatidx_to_featidx(relfeati)
    if featangle[featrelative][relfeati] and (feati not in discreteidx):
      rel2cossinmap.append(np.array([csi,csi+1]))
      csi += 2
    else:
      rel2cossinmap.append(csi)
      csi += 1
  return rel2cossinmap,csi

def relpose_angle_to_cos_sin(relpose,discreteidx=[]):
  """
  relpose_angle_to_cos_sin(relposein)
  convert the relative pose angles features from radians to cos and sin
  relposein: shape = (nrelative,...)
  """
  
  rel2cossinmap,ncs = relfeatidx_to_cossinidx(discreteidx)
  
  relpose_cos_sin = np.zeros((ncs,)+relpose.shape[1:],dtype=relpose.dtype)
  for relfeati in range(nrelative):
    csi = rel2cossinmap[relfeati]
    if type(csi) is int:
      relpose_cos_sin[csi,...] = relpose[relfeati,...]
    else:
      relpose_cos_sin[csi[0],...] = np.cos(relpose[relfeati,...])
      relpose_cos_sin[csi[1],...] = np.sin(relpose[relfeati,...])
  return relpose_cos_sin

def relpose_cos_sin_to_angle(relpose_cos_sin,discreteidx=[],epsilon=1e-6):
  sz = relpose_cos_sin.shape[:-1]
  if len(sz) == 0:
    n = 1
  else:
    n = np.prod(sz)
  relpose_cos_sin = relpose_cos_sin.reshape((n,relpose_cos_sin.shape[-1]))
  rel2cossinmap,ncs = relfeatidx_to_cossinidx(discreteidx)
    
  relpose = np.zeros((n,nrelative),dtype=relpose_cos_sin.dtype)
  for relfeati in range(nrelative):
    csi = rel2cossinmap[relfeati]
    if type(csi) is int:
      relpose[...,relfeati] = relpose_cos_sin[...,csi]
    else:
      # if the norm is less than epsilon, just make the angle 0
      idxgood = np.linalg.norm(relpose_cos_sin[:,csi],axis=-1) >= epsilon
      relpose[idxgood,relfeati] = np.arctan2(relpose_cos_sin[idxgood,csi[1]],relpose_cos_sin[idxgood,csi[0]])
      
  relpose = relpose.reshape(sz+(nrelative,))
  return relpose

def compute_relpose_tspred(relposein,tspred_dct=[],discreteidx=[]):
  """
  compute_relpose_tspred(relpose,tspred_dct=[])
  concatenate the relative pose at t+tau for all tau in tspred_dct
  relposein: shape = (nrelative,T,nflies)
  tspred_dct: list of int
  returns relpose_tspred: shape = (nrelative,T,ntspred_dct+1,nflies)
  """

  ntspred_dct = len(tspred_dct)
  T = relposein.shape[1]
  nflies = relposein.shape[2]
  relpose = relpose_angle_to_cos_sin(relposein,discreteidx=discreteidx)
  nrelrep = relpose.shape[0]

  # predict next frame pose
  relpose_tspred = np.zeros((nrelrep,T,ntspred_dct+1,nflies),dtype=relpose.dtype)
  relpose_tspred[:] = np.nan
  relpose_tspred[:,:-1,0,:] = relpose[:,1:,:]
  for i,toff in enumerate(tspred_dct):
    relpose_tspred[:,:-toff,i+1,:] = relpose[:,toff:,:]
  
  return relpose_tspred


def compute_movement(X=None,scale=None,relpose=None,globalpos=None,simplify=None,
                     dct_m=None,tspred_global=[1,],compute_pose_vel=True,discreteidx=[],
                     returnidx=False,debug=False):
  """
  movement = compute_movement(X=X,scale=scale,...)
  movement = compute_movement(relpose=relpose,globalpos=globalpos,...)

  Args:
      X (ndarray, nkpts x 2 x T x nflies, optional): Keypoints. Can be None only if relpose and globalpos are input. Defaults to None. T>=2
      scale (ndarray, nscale x nflies): Scaling parameters related to an individual fly. Can be None only if relpose and globalpos are input. Defaults to None.
      relpose (ndarray, nrelative x T x nflies or nrelative x T, optional): Relative pose features. T>=2
      If input, X and scale are ignored. Defaults to None.
      globalpos (ndarray, nglobal x T x nflies or nglobal x T, optional): Global position. If input, X and scale are ignored. Defaults to None. T>=2
      simplify (string or None, optional): Whether/how to simplify the output. Defaults to None for no simplification.
  Optional args:
      dct_m (ndarray, nrelative x ntspred_dct+1 x nflies): DCT matrix for pose features. Defaults to None.
      tspred_global (list of int, optional): Time steps to predict for global features. Defaults to [1,].

  Returns:
      movement (ndarray, d_output x T-1 x nflies): Per-frame movement. movement[:,t,i] is the movement from frame 
      t for fly i. 
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

  # centroid and orientation position
  Xorigin = globalpos[:2,...]
  Xtheta = globalpos[2,...]

  # which future frames are we predicting, how many features are there total
  ntspred_global = len(tspred_global)
  if (dct_m is not None) and simplify != 'global':
    ntspred_dct = dct_m.shape[0]
    tspred_dct = np.arange(1,ntspred_dct+1)
    tspred_relative = tspred_dct
  else:
    ntspred_dct = 0
    tspred_dct = []
    tspred_relative = [1,]

  # compute the max of tspred_global and tspred_dct
  tspred_all = np.unique(np.concatenate((tspred_global,tspred_relative)))
  lastT = T - np.max(tspred_all)

  # global velocity  
  # dXoriginrel[tau,:,t,fly] is the global position for fly fly at time t+tau in the coordinate system of the fly at time t
  dXoriginrel,dtheta = compute_global_velocity(Xorigin,Xtheta,tspred_global)

  # relpose_rep is (nrelrep,T,ntspred_dct+1,nflies)
  if simplify == 'global':
    relpose_rep = np.zeros((0,T,ntspred_global+1,nflies),dtype=relpose.dtype)
  elif compute_pose_vel:
    relpose_rep = compute_relpose_velocity(relpose,tspred_dct)
  else:
    relpose_rep = compute_relpose_tspred(relpose,tspred_dct,discreteidx=discreteidx)
  nrelrep = relpose_rep.shape[0]

  if debug:
    # try to reconstruct xorigin, xtheta from dxoriginrel and dtheta
    xtheta0 = Xtheta[0]
    xorigin0 = Xorigin[:,0]
    thetavel = dtheta[0,:]
    xtheta = np.cumsum(np.concatenate((xtheta0[None],thetavel),axis=0),axis=0)
    xoriginvelrel = dXoriginrel[0]
    xoriginvel = mabe.rotate_2d_points(xoriginvelrel.reshape((2,-1)).T,-xtheta[:-1].flatten()).T.reshape(xoriginvelrel.shape)
    xorigin = np.cumsum(np.concatenate((xorigin0[:,None],xoriginvel),axis=1),axis=1)
    print('xtheta0 = %s'%str(xtheta0[0]))
    print('xorigin0 = %s'%str(xorigin0[:,0]))
    print('xtheta[:5] = %s'%str(xtheta[:5,0]))
    print('original Xtheta[:5] = %s'%str(Xtheta[:5,0]))
    print('xoriginvelrel[:5] = \n%s'%str(xoriginvelrel[:,:5,0]))
    print('xoriginvel[:5] = \n%s'%str(xoriginvel[:,:5,0]))
    print('xorigin[:5] = \n%s'%str(xorigin[:,:5,0]))
    print('original Xorigin[:5] = \n%s'%str(Xorigin[:,:5,0]))
    print('max error origin: %e'%np.max(np.abs(xorigin[:,:-1]-Xorigin)))
    print('max error theta: %e'%np.max(np.abs(mabe.modrange(xtheta[:-1]-Xtheta,-np.pi,np.pi))))
    pass

  # only full data up to frame lastT
  # dXoriginrel is (ntspred_global,2,lastT,nflies)
  dXoriginrel = dXoriginrel[:,:,:lastT,:]
  # dtheta is (ntspred_global,lastT,nflies)
  dtheta = dtheta[:,:lastT,:]
  # relpose_rep is (nrelrep,lastT,ntspred_dct+1,nflies)
  relpose_rep = relpose_rep[:,:lastT]

  if (simplify != 'global') and (dct_m is not None):
    # the pose forecasting papers compute error on the actual pose, not the dct. they just force the network to go through the dct
    # representation first.
    relpose_rep[:,:,1:,:] = dct_m @ relpose_rep[:,:,1:,:]
  # relpose_rep is now (ntspred_dct+1,nrelrep,lastT,nflies)
  relpose_rep = np.moveaxis(relpose_rep,2,0)

  if debug and dct_m is not None:
    idct_m = np.linalg.inv(dct_m)
    relpose_rep_dct = relpose_rep[1:].reshape((ntspred_dct,-1))
    relpose_rep_idct = idct_m @ relpose_rep_dct
    relpose_rep_idct = relpose_rep_idct.reshape((ntspred_dct,)+relpose_rep.shape[1:])
    err_dct_0 = np.max(np.abs(relpose_rep_idct[0] - relpose_rep[0]))
    print('max error dct_0: %e'%err_dct_0)
    err_dct_tau = np.max(np.abs(relpose_rep[0,:,ntspred_dct-1:,:] - relpose_rep_idct[-1,:,:-ntspred_dct+1,:]))
    print('max error dct_tau: %e'%err_dct_tau)

  # concatenate the global (dforward, dsideways, dorientation)
  movement_global = np.concatenate((dXoriginrel[:,[1,0]],dtheta[:,None,:,:]),axis=1)
  # movement_global is now (ntspred_global*nglobal,lastT,nflies)
  movement_global = movement_global.reshape((ntspred_global*nglobal,lastT,nflies))
  # relpose_rep is now ((ntspred_dct+1)*nrelrep,lastT,nflies)
  relpose_rep = relpose_rep.reshape(((ntspred_dct+1)*nrelrep,lastT,nflies))

  if nd == 2: # no flies dimension
    movement_global = movement_global[...,0]
    relpose_rep = relpose_rep[...,0]

  # concatenate everything together
  movement = np.concatenate((movement_global,relpose_rep),axis=0)
  
  if returnidx:
    idxinfo = {}
    idxinfo['global'] = [0,ntspred_global*nglobal]
    idxinfo['global_feat_tau'] = unravel_label_index(np.arange(ntspred_global*nglobal),
                                                     dct_m=dct_m,tspred_global=tspred_global,
                                                     nrelrep=nrelrep)
    idxinfo['relative'] = [ntspred_global*nglobal,ntspred_global*nglobal+nrelrep*(ntspred_dct+1)]
    return movement,idxinfo

  return movement

def compute_sensory_wrapper(Xkp,flynum,theta_main=None,returnall=False,returnidx=False):
  
  # other flies positions
  idxother = np.ones(Xkp.shape[-1],dtype=bool)
  idxother[flynum] = False
  Xkp_other = Xkp[:,:,:,idxother]
  
  xeye_main = Xkp[kpeye,0,:,flynum]
  yeye_main = Xkp[kpeye,1,:,flynum]
  xtouch_main = Xkp[kptouch,0,:,flynum]
  ytouch_main = Xkp[kptouch,1,:,flynum]
  xvision_other = Xkp_other[kpvision_other,0,...].transpose((0,2,1))
  yvision_other = Xkp_other[kpvision_other,1,...].transpose((0,2,1))
  xtouch_other = Xkp_other[kptouch_other,0,...].transpose((0,2,1))
  ytouch_other = Xkp_other[kptouch_other,1,...].transpose((0,2,1))
  
  if theta_main is None:
    _,_,theta_main = mabe.body_centric_kp(Xkp[...,[flynum,]])
    theta_main = theta_main[...,0]
  
  otherflies_vision,wall_touch,otherflies_touch = \
    compute_sensory(xeye_main,yeye_main,theta_main+np.pi/2,
                    xtouch_main,ytouch_main,
                    xvision_other,yvision_other,
                    xtouch_other,ytouch_other)
  sensory = np.r_[wall_touch,otherflies_vision]
  idxinfo = {}
  idxoff = 0
  idxinfo['wall_touch'] = [0,wall_touch.shape[0]]
  idxoff += wall_touch.shape[0]
  idxinfo['otherflies_vision'] = [idxoff,idxoff+otherflies_vision.shape[0]]
  idxoff += otherflies_vision.shape[0]
  
  if otherflies_touch is not None:
    sensory = np.r_[sensory,otherflies_touch]
    idxinfo['otherflies_touch'] = [idxoff,idxoff+otherflies_touch.shape[0]]
    idxoff += otherflies_touch.shape[0]
    
  ret = (sensory,)
  if returnall:
    ret = ret + (wall_touch,otherflies_vision,otherflies_touch)
  if returnidx:
    ret = ret + (idxinfo,)

  return ret

def combine_inputs(relpose=None,sensory=None,input=None,labels=None,dim=0):
  if input is None:
    if sensory is None:
      input = relpose
    else:
      input = np.concatenate((relpose,sensory),axis=dim)
  if labels is not None:
    input = np.concatenate((input,labels),axis=dim)
  return input 

def get_dct_matrix(N):
  """ Get the Discrete Cosine Transform coefficient matrix
  Copied from https://github.com/dulucas/siMLPe/blob/main/exps/baseline_h36m/train.py
  Back to MLP: A Simple Baseline for Human Motion Prediction
  Guo, Wen and Du, Yuming and Shen, Xi and Lepetit, Vincent and Xavier, Alameda-Pineda and Francesc, Moreno-Noguer
  arXiv preprint arXiv:2207.01567
  2022
  Args:
      N (int): number of time points

  Returns:
      dct_m: array of shape N x N with the encoding coefficients
      idct_m: array of shape N x N with the inverse coefficients
  """
  dct_m = np.eye(N)
  for k in np.arange(N):
    for i in np.arange(N):
      w = np.sqrt(2 / N)
      if k == 0:
        w = np.sqrt(1 / N)
      dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
  idct_m = np.linalg.inv(dct_m)
  return dct_m, idct_m

def unravel_label_index(idx,dct_m=None,tspred_global=[1,],nrelrep=None,d_output=None,ntspred_relative=1):
  idx = np.array(idx)
  sz = idx.shape
  idx = idx.flatten()
  if dct_m is not None:
    ntspred_relative = dct_m.shape[0]+1
  offrelative = len(tspred_global)*nglobal
  if nrelrep is None:
    if d_output is None:
      nrelrep = nrelative
    else:
      nrelrep = d_output - offrelative
  ftidx = np.zeros((len(idx),2),dtype=int)
  for ii,i in enumerate(idx):
    if i < offrelative:
      tidx,fidx = np.unravel_index(i,(len(tspred_global),nglobal))
      ftidx[ii] = (fidx,tspred_global[tidx])
    else:
      t,fidx = np.unravel_index(i-offrelative,(ntspred_relative,nrelrep))
      # t = 1 corresponds to next frame
      ftidx[ii] = (fidx+nglobal,t+1)
  return ftidx.reshape(sz+(2,))

def ravel_label_index(ftidx,dct_m=None,tspred_global=[1,],nrelrep=None,d_output=None,ntspred_relative=1):

  ftidx = np.array(ftidx)
  sz = ftidx.shape
  assert sz[-1] == 2
  ftidx = ftidx.reshape((-1,2))
  
  idx = np.zeros(ftidx.shape[:-1],dtype=int)

  if dct_m is not None:
    ntspred_relative = dct_m.shape[0]+1
  offrelative = len(tspred_global)*nglobal
  if nrelrep is None:
    if d_output is None:
      nrelrep = nrelative
    else:
      nrelrep = d_output - offrelative
  
  for i,ft in enumerate(ftidx):
    fidx = ft[0]
    t = ft[1]
    isglobal = fidx < nglobal
    if isglobal:
      # t = 1 corresponds to next frame
      tidx = np.nonzero(tspred_global==t)[0][0]
      assert tidx is not None
      idx[i] = np.ravel_multi_index((tidx,fidx),(len(tspred_global),nglobal))
    else:
      # t = 1 corresponds to next frame
      idx[i] = np.ravel_multi_index((t-1,fidx-nglobal),(ntspred_relative,nrelrep))+offrelative
  
  return idx.reshape(sz[:-1])

def compute_features(X,id=None,flynum=0,scale_perfly=None,smush=True,outtype=None,
                     simplify_out=None,simplify_in=None,dct_m=None,tspred_global=[1,],
                     npad=1,compute_pose_vel=True,discreteidx=[],returnidx=False):
  
  res = {}
  
  # convert to relative locations of body parts
  if id is None:
    scale = scale_perfly
  else:
    scale = scale_perfly[:,id]
  
  relpose,globalpos = compute_pose_features(X[...,flynum],scale)
  relpose = relpose[...,0]
  globalpos = globalpos[...,0]
  if npad == 0:
    endidx = None
  else:
    endidx = -npad
  if simplify_in == 'no_sensory':
    sensory = None
    res['input'] = relpose.T
    if returnidx:
      idxinfo['input'] = {}
      idxinfo['input']['relpose'] = [0,relpose.shape[0]]
  else:
    out = compute_sensory_wrapper(X[:,:,:endidx,:],flynum,theta_main=globalpos[featthetaglobal,:endidx],
                                  returnall=True,returnidx=returnidx)
    sensory,wall_touch,otherflies_vision,otherflies_touch = out[:4]
    if returnidx:
      idxinfo = {}
      idxinfo['input'] = out[4]

    res['input'] = combine_inputs(relpose=relpose[:,:endidx],sensory=sensory).T
    if returnidx:
      idxinfo['input'] = {k: [vv+relpose.shape[0] for vv in v] for k,v in idxinfo['input'].items()}
      idxinfo['input']['relpose'] = [0,relpose.shape[0]]

  out = compute_movement(relpose=relpose,globalpos=globalpos,simplify=simplify_out,dct_m=dct_m,
                         tspred_global=tspred_global,compute_pose_vel=compute_pose_vel,
                         discreteidx=discreteidx,returnidx=returnidx)
  if returnidx:
    movement,idxinfo['labels'] = out
  else:
    movement = out
  
  if simplify_out is not None:
    if simplify_out == 'global':
      movement = movement[featglobal,...]
    else:
      raise
    
  res['labels'] = movement.T
  res['init'] = globalpos[:,:2]
  res['scale'] = scale
  #res['nextinput'] = input[-1,:]
  
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
      if otherflies_touch is not None:
        res['otherflies_touch'] = otherflies_touch[:,:-1]
        res['next_otherflies_touch'] = otherflies_touch[:,-1]
            
  # debug_plot_compute_features(X,porigin,theta,Xother,Xnother)
    
  if outtype is not None:
    res = {key: val.astype(outtype) for key,val in res.items()}
    
  if returnidx:
    return res,idxinfo
  else:
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

def compute_otherflies_touch_mult(data,prct=99):
  
  # 1/maxd^exp = mult*maxd^exp
  
  # X is nkeypts x 2 x T x nflies
  nkpts = data['X'].shape[0]
  T = data['X'].shape[2]
  nflies = data['X'].shape[3]
  # isdata is T x nflies
  # X will be nkpts x 2 x N
  X = data['X'].reshape([nkpts,2,T*nflies])[:,:,data['isdata'].flatten()]
  # maximum distance from some keypoint to any keypoint included in kpother
  d = np.sqrt(np.nanmax(np.nanmin(np.sum((X[None,kptouch,:,:] - X[kptouch_other,None,:,:])**2.,axis=2),axis=0),axis=0))
  maxd = np.percentile(d,prct)

  otherflies_touch_mult = 1./((maxd)**SENSORY_PARAMS['otherflies_touch_exp'])
  return otherflies_touch_mult
  
    
def debug_plot_compute_features(X,porigin,theta,Xother,Xnother):
  
  t = 0
  rplot = 5.
  plt.clf()
  ax = plt.subplot(1,2,1)
  mabe.plot_flies(X[:,:,t,:],ax=ax,textlabels='fly',colors=np.zeros((X.shape[-1],3)))
  ax.plot(porigin[0,t],porigin[1,t],'rx',linewidth=2)
  ax.plot([porigin[0,t,0],porigin[0,t,0]+np.cos(theta[t,0])*rplot],
          [porigin[1,t,0],porigin[1,t,0]+np.sin(theta[t,0])*rplot],'r-')
  ax.plot(Xother[kpvision_other,0,t,:],Xother[kpvision_other,1,t,:],'o')
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

def to_size(sz):
  if sz is None:
    sz = (1,)
  elif isinstance(sz,int):
    sz = (sz,)
  elif isinstance(sz,list):
    sz = tuple(sz)
  elif isinstance(sz,tuple):
    pass
  else:
    raise ValueError('Input sz must be an int, list, or tuple')
  return sz

def weighted_sample(w,nsamples=0):
  SMALLNUM = 1e-6
  assert(torch.all(w>=0.))
  nbins = w.shape[-1]
  szrest = w.shape[:-1]
  n = int(np.prod(szrest))
  p = torch.cumsum(w.reshape((n,nbins)),dim=-1)
  assert(torch.all(torch.abs(p[:,-1]-1)<=SMALLNUM))
  p[p>1.] = 1.
  p[:,-1] = 1.
  if nsamples == 0:
    nsamples1 = 1
  else:
    nsamples1 = nsamples
  r = torch.rand((nsamples1,n),device=w.device)
  s = torch.zeros((nsamples1,)+p.shape,dtype=w.dtype,device=w.device)
  s[:] = r[...,None] <= p
  idx = torch.argmax(s,dim=-1)
  if nsamples > 0:
    szrest = (nsamples,)+szrest
  return idx.reshape(szrest)

# def samplebin(X,sz=None):
#   sz = to_size(sz)
#   r = torch.randint(low=0,high=X.numel(),size=sz)
#   return X[r]

def select_bin_edges(movement,nbins,bin_epsilon,outlierprct=0,feati=None):

  n = movement.shape[0]
  lims = np.percentile(movement,[outlierprct,100-outlierprct])
  max_bin_epsilon = (lims[1]-lims[0])/(nbins+1)
  if bin_epsilon >= max_bin_epsilon:
    print(f'{feati}: bin_epsilon {bin_epsilon} bigger than max bin epsilon {max_bin_epsilon}, setting all bins to be the same size')
    bin_edges = np.linspace(lims[0],lims[1],nbins+1)
    return bin_edges

  bin_edges = np.arange(lims[0],lims[1],bin_epsilon)
  bin_edges[-1] = lims[1]
  
  counts,_ = np.histogram(movement,bin_edges)
  mergecounts = counts[1:]+counts[:-1]
  for iter in range(len(bin_edges)-nbins-1):
    mincount = np.min(mergecounts)
    bini = np.random.choice(np.nonzero(mergecounts==mincount)[0],1)[0]
    if bini > 0:
      mergecounts[bini-1] += counts[bini]
    if bini < len(mergecounts)-1:
      mergecounts[bini+1] += counts[bini]
    mergecounts = np.delete(mergecounts,bini)
    counts[bini] = mincount
    counts = np.delete(counts,bini+1)
    bin_edges = np.delete(bin_edges,bini+1)

  return bin_edges
  

def fit_discretize_labels(data,featidx,nbins=50,bin_epsilon=None,outlierprct=.001,fracsample=None,nsamples=None):

  # compute percentiles
  nfeat = len(featidx)
  prctiles_compute = np.linspace(0,100,nbins+1)
  prctiles_compute[0] = outlierprct
  prctiles_compute[-1] = 100-outlierprct
  movement = np.concatenate([example['labels'][:,featidx] for example in data],axis=0)
  dtype = movement.dtype

  # bin_edges is nfeat x nbins+1
  if bin_epsilon is not None:
    bin_edges = np.zeros((nfeat,nbins+1),dtype=dtype)
    for feati in range(nfeat):
      bin_edges[feati,:] = select_bin_edges(movement[:,feati],nbins,bin_epsilon[feati],outlierprct=outlierprct,feati=feati)
  else:
    bin_edges = np.percentile(movement,prctiles_compute,axis=0)
    bin_edges = bin_edges.astype(dtype).T

  binnum = np.zeros(movement.shape,dtype=int)
  for i in range(nfeat):
    binnum[:,i] = np.digitize(movement[:,i],bin_edges[i,:])
  binnum = np.minimum(np.maximum(0,binnum-1),nbins-1)

  if nsamples is None:
    if fracsample is None:
      fracsample = 1/nbins/5
    nsamples = int(np.round(fracsample*movement.shape[0]))

  # for each bin, approximate the distribution
  samples = np.zeros((nsamples,nfeat,nbins),movement.dtype)
  bin_means = np.zeros((nfeat,nbins),movement.dtype)
  bin_medians = np.zeros((nfeat,nbins),movement.dtype)
  for i in range(nfeat):
    for j in range(nbins):
      movementcurr = torch.tensor(movement[binnum[:,i]==j,i])
      if movementcurr.shape[0] == 0:
        bin_means[i,j] = (bin_edges[i,j]+bin_edges[i,j+1])/2.
        bin_medians[i,j] = bin_means[i,j]
        samples[:,i,j] = bin_means[i,j]
      else:
        samples[:,i,j] = np.random.choice(movementcurr,size=nsamples,replace=True)
        bin_means[i,j] = np.nanmean(movementcurr)
        bin_medians[i,j] = np.nanmedian(movementcurr)

      #kde[j,i] = KernelDensity(kernel='tophat',bandwidth=kde_bandwidth).fit(movementcurr[:,None])

  return bin_edges,samples,bin_means,bin_medians

def discretize_labels(movement,bin_edges,soften_to_ends=False):

  n = movement.shape[0]
  nfeat = bin_edges.shape[0]
  nbins = bin_edges.shape[1]-1

  bin_centers = (bin_edges[:,1:]+bin_edges[:,:-1])/2.
  bin_width = (bin_edges[:,1:]-bin_edges[:,:-1])

  #d = np.zeros((n,nbins+1))
  labels = np.zeros((n,nfeat,nbins),dtype=movement.dtype)#,dtype=bool)
  if soften_to_ends:
    lastbin = 0
  else:
    lastbin = 1
    
  for i in range(nfeat):
    binnum = np.digitize(movement[:,i],bin_edges[i,:])
    binnum = np.minimum(nbins-1,np.maximum(0,binnum-1))
    # soft binning
    # don't soften into end bins
    idxsmall = (movement[:,i] < bin_centers[i,binnum]) & (binnum > lastbin)
    idxlarge = (movement[:,i] > bin_centers[i,binnum]) & (binnum < (nbins-1-lastbin))
    idxedge = (idxsmall==False) & (idxlarge==False)
    # distance from bin center, max should be .5
    d = (np.abs(movement[:,i]-bin_centers[i,binnum]) / bin_width[i,binnum])
    d[idxedge] = 0.
    labels[np.arange(n),i,binnum] = 1. - d
    labels[idxsmall,i,binnum[idxsmall]-1] = d[idxsmall]
    labels[idxlarge,i,binnum[idxlarge]+1] = d[idxlarge]
    
    #d[:,-1] = True
    #d[:,1:-1] = movement[:,i,None] <= bin_edges[None,1:-1,i]
    #labels[:,:,i] = (d[:,:-1] == False) & (d[:,1:] == True)

  return labels

def labels_discrete_to_continuous(labels_discrete,bin_edges):

  sz = labels_discrete.shape
  
  nbins = sz[-1]
  nfeat = sz[-2]
  szrest = sz[:-2]
  n = np.prod(np.array(szrest))
  labels_discrete = torch.reshape(labels_discrete,(n,nfeat,nbins))  
  # nfeat x nbins
  bin_centers = (bin_edges[:,1:]+bin_edges[:,:-1])/2.
  s = torch.sum(labels_discrete,dim=-1)
  assert torch.max(torch.abs(1-s)) < .01, 'discrete labels do not sum to 1'
  movement = torch.sum(bin_centers[None,...]*labels_discrete,dim=-1) / s
  movement = torch.reshape(movement,szrest+(nfeat,))

  return movement


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

class ObservationInputs:

  def __init__(self,example_in=None,Xkp=None,fly=0,scale=None,dataset=None,dozscore=False,**kwargs):
    
    # to do: deal with flattening
    
    self.input = None
    self.init = None
    self.metadata = None
    self.d_input = None
  
    self.set_params(kwargs)
    if dataset is not None:
      self.set_params(self.get_params_from_dataset(dataset),override=False)
    default_params = FlyExample.get_default_params()
    self.set_params(default_params,override=False)
    
    self.sensory_feature_idx,self.sensory_feature_szs = \
      get_sensory_feature_shapes(simplify=self.simplify_in)      
    if example_in is not None:
      self.set_example(example_in,dozscore=dozscore)
    elif Xkp is not None:
      self.set_inputs_from_keypoints(Xkp,fly,scale)

    if self.flatten_obs_idx is not None:
      self.flatten_max_dinput = np.max(list(self.flatten_obs_idx.values()))
    else:
      self.flatten_dinput_pertype = np.array(self.d_input)
      self.flatten_max_dinput = self.d_input

    if self.flatten_obs:
      flatten_dinput_pertype = np.array([v[1]-v[0]] for v in self.flatten_obs_idx.values())
      self.flatten_input_type_to_range = np.zeros((self.flatten_dinput_pertype.size,2),dtype=int)
      cs = np.cumsum(flatten_dinput_pertype)
      self.flatten_input_type_to_range[1:,0] = cs[:-1]
      self.flatten_input_type_to_range[:,1] = cs
    
    return

  def update_sizes(self):
    self.d_input = self.input.shape[-1]
    self.pre_sz = self.input.shape[:-2]
  
  @staticmethod
  def flyexample_to_observationinput_params(params):
    zscore_params_input,_ = FlyExample.split_zscore_params(params['zscore_params'])
    kwinputs = {'zscore_params':zscore_params_input,}
    if 'simplify_in' in params:
      kwinputs['simplify_in'] = params['simplify_in']
    if 'flatten_obs' in params:
      kwinputs['flatten_obs'] = params['flatten_obs']
    if 'starttoff' in params:
      kwinputs['starttoff'] = params['starttoff']
    if 'flatten_obs_idx' in params:
      kwinputs['flatten_obs_idx'] = params['flatten_obs_idx']
    if 'do_input_labels' in params:
      kwinputs['do_input_labels'] = params['do_input_labels']
    return kwinputs
  
  @staticmethod
  def get_default_params():
    params = FlyExample.get_default_params()
    params = ObservationInputs.flyexample_to_observationinput_params(params)
    return params
  
  def get_params(self):
    params = {
      'zscore_params':self.zscore_params,
      'simplify_in':self.simplify_in,
      'flatten_obs':self.flatten_obs,
    }
    return params
  
  def set_params(self,params,override=True):
    for k,v in params.items():
      if override or (not hasattr(self,k)) or (getattr(self,k) is None):
        setattr(self,k,v)
  
  @staticmethod
  def get_params_from_dataset(dataset):
    params = FlyExample.get_params_from_dataset(dataset)
    params = ObservationInputs.flyexample_to_observationinput_params(params)
    return params
  
  def set_example(self,example_in,dozscore=False):
    self.input = example_in['input']
    if 'init_all' in example_in:
      self.init = example_in['init_all']
    else:
      self.init = example_in['init']
    if 'input_init' in example_in:
      self.input = np.concatenate((example_in['input_init'],self.input),axis=-2)
      
    self.update_sizes()
    
    if dozscore:
      self.input = zscore(self.input,self.zscore_params['mu_input'],self.zscore_params['sig_input'])
    
    if 'metadata' in example_in:
      self.metadata = example_in['metadata']

  @property
  def ntimepoints(self):
    if self.input is None:
      return 0
    return self.input.shape[-2]
  
  def is_zscored(self):
    return self.zscore_params is not None
  
  def get_raw_inputs(self,makecopy=True):
    if makecopy:
      return self.input.copy()
    else:
      return self.input
            
  def get_inputs(self,zscored=False,**kwargs):
    input = self.get_raw_inputs(**kwargs)

    # todo: deal with flattening
    if self.is_zscored() and zscored == False:
      input = unzscore(input,self.zscore_params['mu_input'],self.zscore_params['sig_input'])
    
    return input
    
  def get_init_next(self,**kwargs):

    if self.init is None:
      return None

    tau = self.init.shape[-1]
    szrest = self.init.shape[:-2]
    relative0 = self.get_inputs_type('pose',zscored=False,**kwargs)[...,:tau,:]
    global0 = self.init

    next0 = np.zeros(szrest+(nfeatures,tau),dtype=relative0.dtype)
    next0[...,featglobal,:] = global0
    next0[...,featrelative,:] = np.moveaxis(relative0,-1,-2)
      
    return next0
  
  def get_split_inputs(self,**kwargs):
    input = self.get_inputs(**kwargs)
    input = split_features(input)
    return input
  
  def get_inputs_type(self,type,**kwargs):
    input = self.get_split_inputs(**kwargs)
    return input[type]
  
  def set_zscore_params(self,zscore_params):
    self.zscore_params = zscore_params
    return
  
  def add_pose_noise(self,train_inputs,eta_pose,zscored=False):
    idx = self.sensory_feature_idx['pose']
    if self.is_zscored() and (zscored==False):
      eta_pose = zscore(eta_pose,self.zscore_params['mu_input'][...,idx[0]:idx[1]],
                        self.zscore_params['sig_input'][...,idx[0]:idx[1]])
    train_inputs[...,idx[0]:idx[1]] += eta_pose
    return train_inputs

  def set_inputs(self,input,zscored=False):
    if zscored == False and self.is_zscored():
      input = zscore(input,self.zscore_params['mu_input'],self.zscore_params['sig_input'])
    self.input = input
    return
  
  def append_inputs(self,toappend,zscored=False):
    if zscored == False and self.is_zscored():
      toappend = zscore(input,self.zscore_params['mu_input'],self.zscore_params['sig_input'])
    self.input = np.concatenate((self.input,toappend),axis=-2)
    return

  def set_inputs_from_keypoints(self,Xkp,fly,scale=None):
    sensory = compute_sensory_wrapper(Xkp,fly)
    labels = PoseLabels(Xkp=Xkp[...,fly],scale=scale,is_velocity=False)
    pose_relative = labels.get_next_pose_relative()
    input = combine_inputs(relpose=pose_relative,sensory=sensory,dim=-1)
    self.set_inputs(input,zscored=False)
    return
  
  def append_keypoints(self,Xkp,fly,scale=None):
    sensory = compute_sensory_wrapper(Xkp,fly)
    labels = PoseLabels(Xkp=Xkp[...,fly],scale=scale,is_velocity=False)
    pose_relative = labels.get_next_pose_relative()
    input = combine_inputs(relpose=pose_relative,sensory=sensory,dim=-1)
    self.append_inputs(input,zscored=False)
    return
  
  def add_noise(self,train_inputs,input_labels=None,labels=None):
    assert(np.prod(self.pre_sz) == 1)
    T = self.ntimepoints
    if input_labels is not None:
      d_labels = input_labels.shape[-1]
    else:
      d_labels = self.sensory_feature.szs['pose']
    # additive noise
    eta = np.zeros((T,d_labels))
    do_add_noise = np.random.rand(T) <= self.p_add_input_noise
    eta[do_add_noise,:] = self.input_noise_sigma[None,:]*np.random.randn(np.count_nonzero(do_add_noise),self.d_output)
    if input_labels is None:
      eta_pose = eta
    else:
      eta_input_labels = labels.next_to_input_labels(eta)
      eta_pose = labels.next_to_nextpose(eta)
      input_labels += eta_input_labels
    train_inputs = self.add_pose_noise(train_inputs,eta_pose)
    return train_inputs,input_labels,eta
  
  
  def get_sensory_feature_idx(self):
    return get_sensory_feature_idx(self.simplify_in)
    
  def get_train_inputs(self,input_labels=None,do_add_noise=False,labels=None):

    train_inputs = self.get_raw_inputs()
    
    # makes a copy
    if do_add_noise:
      train_inputs,input_labels,eta = self.add_noise(train_inputs,input_labels,labels)
    else:
      eta = None

    # makes a copy
    train_inputs = torch.tensor(train_inputs)
    train_inputs_init = None
    
    if not self.flatten_obs:
      if input_labels is not None:
        train_inputs_init = train_inputs[...,:self.starttoff,:]
        train_inputs = torch.cat((torch.tensor(input_labels[...,:-self.starttoff,:]),
                                  train_inputs[...,self.starttoff:,:]),dim=-1)
    else:
      ntypes = len(self.flatten_obs_idx)
      flatinput = torch.zeros(self.pre_sz+(self.ntimepoints,ntypes,self.flatten_max_dinput),dtype=train_inputs.dtype)
      for i,v in enumerate(self.flatten_obs_idx.values()):
        flatinput[...,i,self.flatten_input_type_to_range[i,0]:self.flatten_input_type_to_range[i,1]] = train_inputs[:,v[0]:v[1]]

      train_inputs = flatinput

    return {'input': train_inputs,'eta': eta, 'input_init': train_inputs_init}

  
class PoseLabels:
  def __init__(self,example_in=None,init_next=None,
               Xkp=None,scale=None,metadata=None,
               dozscore=False,dodiscretize=False,
               dataset=None,**kwargs):
    
    # different representations of labels:
    # labels_raw -- representation used for training/prediction
    # store in this format so that it is efficient for training
    # this contains the follow:
    # continuous: (sz) x d_output_continuous
    # discrete: (sz) x d_output_discrete x nbins
    # todiscretize: (sz) x d_output_discrete
    # stacked: (sz) x ntypes x d_output_flatten
    # these will be z-scored if zscore_params is not None
    # 
    # full_labels_discreteidx: indices of 

    self.set_params(kwargs)
    if dataset is not None:
      self.set_params(self.get_params_from_dataset(dataset),override=False)
    default_params = PoseLabels.get_default_params()
    self.set_params(default_params,override=False)

    # default_params = self.get_default_params()
    # self.set_params(default_params,override=False)

    # copy over labels_in
    self.label_keys = {}
    self.labels_raw = {}
    self.pre_sz = None
    self.metadata = metadata
    self.categories = None
        
    if (self.discretize_params is not None) and ('bin_edges' in self.discretize_params) \
      and (self.discretize_params['bin_edges'] is not None):
      self.discretize_nbins = self.discretize_params['bin_edges'].shape[-1]-1
    else:
      self.discretize_nbins = 0

    # to do: include flattening

    self.init_pose = init_next
    
    if example_in is not None:
      self.set_raw_example(example_in,dozscore=dozscore,dodiscretize=dodiscretize)
    elif Xkp is not None:
      self.set_keypoints(Xkp,scale)

    if 'continuous' in self.labels_raw:
      assert self.d_multicontinuous == self.labels_raw['continuous'].shape[-1]
    if self.is_discretized() and 'discrete' in self.labels_raw:
      assert self.d_multidiscrete == self.labels_raw['discrete'].shape[-2]
    
    return
  
  def __str__(self):
    s = f'PoseLabels:\n'
    if len(self.labels_raw) == 0:
      s += 'No data set'
      return s
    s += f'  pre size: {self.pre_sz}\n'
    s += f'  ntimepoints: {self.ntimepoints}\n'
    if self.is_continuous():
      s += f'  continuous dim: {self.labels_raw["continuous"].shape[-1]}\n'
    if self.is_discretized():
      s += f'  discrete dim: {self.labels_raw["discrete"].shape[-2]}\n'
      s += f'  nbins: {self.labels_raw["discrete"].shape[-1]}\n'
    return s
  
  def set_prediction(self,pred):
    if self.is_continuous():
      if 'continuous' in pred:
        self.labels_raw['continuous'][...,self.starttoff:,:] = pred['continuous']
      elif 'labels' in pred:
        self.labels_raw['continuous'][...,self.starttoff:,:] = pred['labels']
      else:
        raise ValueError('pred must contain continuous or labels')
    if self.is_discretized():
      if 'discrete' in pred:
        self.labels_raw['discrete'][...,self.starttoff:,:,:] = pred['discrete']
      elif 'labels_discrete' in pred:
        self.labels_raw['discrete'][...,self.starttoff:,:,:] = pred['labels_discrete']
      else:
        raise ValueError('pred must contain discrete or labels_discrete')
    return
  
  def set_raw_example(self,example_in,dozscore=False,dodiscretize=False):

    if example_in is None:
      self.labels_raw = {}
      self.label_keys = {}
      self.metadata = None
      self.categories = None
      self.pre_sz = None
      self.scale = None
      return
    
    if 'labels' in example_in:
      labels_in = example_in['labels']
      self.label_keys['continuous'] = 'labels'
    elif 'continuous' in example_in:
      labels_in = example_in['continuous']
      self.label_keys['continuous'] = 'continuous'
    else:
      raise ValueError('labels_in must contain labels or continuous')
    tinit = 0
    if 'labels_init' in example_in and example_in['labels_init'] is not None:
      labels_in = np.concatenate((example_in['labels_init'],labels_in),axis=-2)
      tinit = example_in['labels_init'].shape[-2]
    elif 'continuous_init' in example_in and example_in['continuous_init'] is not None:
      labels_in = np.concatenate((example_in['continuous_init'],labels_in),axis=-2)
      tinit = example_in['continuous_init'].shape[-2]
    self.labels_raw['continuous'] = np.atleast_2d(labels_in)

    if 'labels_discrete' in example_in:
      labels_discrete = example_in['labels_discrete']      
      self.label_keys['discrete'] = 'labels_discrete'
    elif 'discrete' in example_in:
      labels_discrete = example_in['discrete']
      self.label_keys['discrete'] = 'discrete'
    else:
      labels_discrete = None
    if labels_discrete is not None:
      labels_discrete = np.atleast_3d(labels_discrete)
      if 'labels_discrete_init' in example_in and example_in['labels_discrete_init'] is not None:
        labels_discrete = np.concatenate((example_in['labels_discrete_init'],labels_discrete),axis=-3)
      elif 'discrete_init' in example_in and example_in['discrete_init'] is not None:
        labels_discrete = np.concatenate((example_in['discrete_init'],labels_discrete),axis=-3)
      self.labels_raw['discrete'] = labels_discrete

    if 'labels_todiscretize' in example_in:
      labels_todiscretize = example_in['labels_todiscretize']
      self.label_keys['todiscretize'] = 'labels_todiscretize'
    elif 'todiscretize' in example_in:
      labels_todiscretize = example_in['todiscretize']
      self.label_keys['todiscretize'] = 'todiscretize'
    else:
      labels_todiscretize = None
    if labels_todiscretize is not None:
      labels_todiscretize = np.atleast_2d(labels_todiscretize)
      if 'labels_todiscretize_init' in example_in and example_in['labels_todiscretize_init'] is not None:
        labels_todiscretize = np.concatenate((example_in['labels_todiscretize_init'],labels_todiscretize),axis=-2)
      elif 'todiscretize_init' in example_in and example_in['todiscretize_init'] is not None:
        labels_todiscretize = np.concatenate((example_in['todiscretize_init'],labels_todiscretize),axis=-2)
      self.labels_raw['todiscretize'] = labels_todiscretize
      
    if self.is_continuous():
      self.pre_sz = self.labels_raw['continuous'].shape[:-2]
    else:
      self.pre_sz = self.labels_raw['discrete'].shape[:-3]
      
    if 'mask' in example_in:
      self.labels_raw['mask'] = np.atleast_1d(example_in['mask'])
      if tinit > 0:
        self.labels_raw['mask'] = np.concatenate((np.zeros(self.pre_sz+(tinit,),dtype=bool),self.labels_raw['mask']),axis=-1)
      
    # if 'labels_stacked' in example_in:
    #   self.labels_raw['stacked'] = example_in['labels_stacked']
    #   self.label_keys['stacked'] = 'labels_stacked'
    # elif 'stacked' in example_in:
    #   self.labels_raw['stacked'] = example_in['stacked']
    #   self.label_keys['stacked'] = 'stacked'
    self.scale = example_in['scale']
    if 'metadata' in example_in:
      self.metadata = example_in['metadata']
      
    if 'categories' in example_in:
      self.categories = example_in['categories']
      
    if dozscore and self.is_zscored():
      self.labels_raw['continuous'] = self.zscore_multi(self.labels_raw['continuous'])
    if dodiscretize and self.is_discretized():
      self.discretize_multi(self.labels_raw)
  
  def append_raw(self,pred):
    if 'labels' in pred:
      toappend = np.atleast_2d(pred['labels'])
    elif 'continuous' in pred:
      toappend = np.atleast_2d(pred['continuous'])
    else:
      raise ValueError('pred must contain labels or continuous')
    tappend = toappend.shape[-2]
    self.labels_raw['continuous'].append(toappend,axis=-2)
    if 'discrete' in self.labels_raw:
      if 'labels_discrete' in pred:
        toappend = np.atleast_2d(pred['labels_discrete'])
      elif 'discrete' in pred:
        toappend = np.atleast_2d(pred['discrete'])
      else:
        raise ValueError('pred must contain labels_discrete or discrete')
      assert toappend.shape[-2] == tappend
      self.labels_raw['discrete'].append(toappend,axis=-2)

    if 'todiscretize' in self.labels_raw:
      if 'labels_todiscretize' in pred:
        toappend = np.atleast_2d(pred['labels_todiscretize'])
      elif 'todiscretize' in pred:
        toappend = np.atleast_2d(pred['todiscretize'])
      else:
        toappend = np.zeros(self.pre_sz+(tappend,self.d_multidiscrete),self.dtype)
        toappend[:] = np.nan
      self.labels_raw['todiscretize'].append(toappend,axis=-2)
      
    return
  
  def copy(self):
    return self.copy_subindex()
  
  def copy_subindex(self,idx_pre=None,ts=None):
    
    labels = self.get_raw_labels(makecopy=True)
    labels['metadata'] = self.get_metadata(makecopy=True)
    init_next = self.get_init_pose()
    
    if idx_pre is not None:
      ks = ['continuous', 'discrete', 'todiscretize','init','scale','categories','mask']
      for k in ks:
        if k in labels:
          labels[k] = labels[k][idx_pre]
      for k in labels['metadata'].keys():
        labels['metadata'][k] = labels['metadata'][k][idx_pre]
      init_next = init_next[idx_pre]
        
    if ts is not None:
      # hasn't been tested yet...
      ks = ['continuous', 'discrete', 'todiscretize','mask']
      if 'categories' in labels:
        cattextra = labels['categories'].shape[-1] - labels['continuous'].shape[-2]
      if hasattr(ts,'__len__'):
        assert np.all(np.diff(ts) == 1), 'ts must be consecutive'
        toff = ts[0]
        if 'categories' in labels:
          labels['categories'] = labels['categories'][...,ts[0]:ts[-1]+cattextra,:]
      else:
        toff = ts
        if 'categories' in labels:
          labels['categories'] = labels['categories'][...,ts:ts+cattextra,:]
      for k in ks:
        if k not in labels:
          continue
        if k == 'discrete':
          labels[k] = labels[k][...,ts,:,:]
        else:
          labels[k] = labels[k][...,ts,:]
      labels['metadata']['t0'] += toff
      
    new = PoseLabels(example_in=labels,init_next=init_next,**self.get_params())
    return new
  
  def erase_labels(self):
    if self.is_continuous() and 'continuous' in self.labels_raw:
      self.labels_raw['continuous'][...,self.starttoff:,:] = np.nan
    if self.is_discretized():
      if 'discrete' in self.labels_raw:
        self.labels_raw['discrete'][...,self.starttoff:,:,:] = np.nan
      if 'todiscretize' in self.labels_raw:
        self.labels_raw['todiscretize'][...,self.starttoff:,:] = np.nan
    return
  
  @staticmethod
  def flyexample_to_poselabels_params(params):
    if 'zscore_params' in params:
      _,zscore_params_labels = FlyExample.split_zscore_params(params['zscore_params'])
      params['zscore_params'] = zscore_params_labels
    toremove = ['do_input_labels','flatten_obs','simplify_in','flatten_obs_idx']      
    for k in toremove:
      if k in params:
        del params[k]
    return params
  
  @staticmethod
  def get_default_params():
    params = FlyExample.get_default_params()
    params = PoseLabels.flyexample_to_poselabels_params(params)
    return params
  
  def get_params_from_dataset(self,dataset):
    params = FlyExample.get_params_from_dataset(dataset)
    params = PoseLabels.flyexample_to_poselabels_params(params)
    return params

  def get_params(self):
    kwlabels = {
      'zscore_params':self.zscore_params,
      'discreteidx':self.idx_nextdiscrete_to_next,
      'tspred_global':self.tspred_global,
      'discrete_tspred':self.discrete_tspred,
      'ntspred_relative':self.ntspred_relative,
      'discretize_params':self.discretize_params,
      'is_velocity':self.is_velocity,
      'simplify_out':self.simplify_out,
      'starttoff':self.starttoff,
      'flatten_labels':self.flatten_labels,
      'dct_m':self.dct_m,
      'idct_m':self.idct_m,
      }
    return kwlabels
  
  def set_params(self,params,override=True):
    translatedict = {'discreteidx':'idx_nextdiscrete_to_next'}
    for k,v in params.items():
      if k in translatedict:
        k = translatedict[k]
      if override or (not hasattr(self,k)) or (getattr(self,k) is None):
        setattr(self,k,v)
  
  @property
  def ntimepoints(self):
    # number of time points
    if len(self.labels_raw) == 0:
      return 0
    if self.is_continuous():
      return self.labels_raw['continuous'].shape[-2]
    else:
      return self.labels_raw['discretized'].shape[-3]
  @property
  def ntimepoints_train(self):
    return self.ntimepoints-self.starttoff
  
  @property
  def device(self):
    return self.labels_raw['continuous'].device
  
  @property
  def dtype(self):
    if self.is_continuous():
      return self.labels_raw['continuous'].dtype
    else:
      return self.labels_raw['discretized'].dtype
  
  @property
  def shape(self):
    return self.pre_sz + (self.ntimepoints,self.get_d_labels_full(),)
  
  @property
  def d_labels_full(self):
    return self.d_multi
  
  def is_dct(self):
    return self.ntspred_relative > 1
  
  def set_init_pose(self,init_pose):
    self.init_pose = init_pose
    
  def get_init_pose(self,starttoff=None):
    if starttoff is None:
      return self.init_pose
    else:
      return self.init_pose[:,starttoff]
    
  def get_init_global(self,starttoff=None,makecopy=True):
    init_global0 = self.init_pose[...,self.idx_nextglobal_to_next,:]
    if starttoff is None:
      init_global = init_global0
      if makecopy:
        init_global = init_global.copy()
      else:
        init_global = init_global0
      return init_global
    init_global = init_global0[...,starttoff]
    if makecopy:
      init_global = init_global.copy()
      init_global0 = init_global0.copy()
    return init_global,init_global0
    
  def get_scale(self,makecopy=True):
    if makecopy:
      return self.scale.copy()
    else:
      return self.scale
  
  def get_categories(self,makecopy=True):
    if makecopy:
      return self.categories.copy()
    else:
      return self.categories
  
  def get_metadata(self,makecopy=True):
    if makecopy:
      return copy.deepcopy(self.metadata)
    else:
      return self.metadata
  
  def get_d_labels_input(self):
    return self.d_next_cossin
  
  # which indices of pose (next frame, global + relative) are global
  @property
  def idx_nextglobal_to_next(self):
    return np.array(featglobal)
  @property
  def d_next_global(self):
    return len(self.idx_nextglobal_to_next)

  # which indices of pose (next frame, global + relative) are global
  @property
  def idx_nextglobal_to_next(self):
    return np.array(featglobal)
  @property
  def d_next_global(self):
    return len(self.idx_nextglobal_to_next)

  # which indices of pose (next frame, global + relative) are relative
  @property
  def idx_nextrelative_to_next(self):
    if self.simplify_out is None:
      return np.nonzero(featrelative)[0]
    else:
      return np.array([])
  @property
  def d_next_relative(self):
    return len(self.idx_nextrelative_to_next)

  @property
  def d_next(self):
    return self.d_next_global + self.d_next_relative

  # which indices are angles
  @property
  def is_angle_next(self):
    return featangle
  
  # which indices of pose (next frame, global + relative) are continuous
  @property
  def idx_nextcontinuous_to_next(self):
    iscontinuous = np.ones(self.d_next,dtype=bool)
    iscontinuous[self.idx_nextdiscrete_to_next] = False
    return np.nonzero(iscontinuous)[0]

  # we will use a cosine/sine representation for relative pose
  # next_cossin is equivalent to next if velocity is used
  @property
  def idx_nextcossinglobal_to_nextcossin(self):
    return np.arange(self.d_next_global)
  @property
  def d_next_cossin_global(self):
    return len(self.idx_nextcossinglobal_to_nextcossin)

  @property
  def idx_nextglobal_to_nextcossinglobal(self):
    return np.arange(self.d_next_global)

  def get_idx_nextrelative_to_nextcossinrelative(self):
    if self.is_velocity:
      return np.arange(self.d_next_relative),self.d_next_relative
    else:
      return relfeatidx_to_cossinidx(self.idx_nextdiscrete_to_next)

  @property
  def idx_nextrelative_to_nextcossinrelative(self):
    idx,_ = self.get_idx_nextrelative_to_nextcossinrelative()
    return idx
  @property
  def d_next_cossin_relative(self):
    _,d = self.get_idx_nextrelative_to_nextcossinrelative()
    return d

  @property
  def d_next_cossin(self):
    return self.d_next_cossin_relative + self.d_next_cossin_global

  @property
  def idx_nextcossinrelative_to_nextcossin(self):
    return np.setdiff1d(np.arange(self.d_next_cossin),self.idx_nextcossinglobal_to_nextcossin)

  @property
  def idx_next_to_nextcossin(self):
    idx = list(range(self.d_next))
    idx_nextglobal_to_next = self.idx_nextglobal_to_next
    idx_nextglobal_to_nextcossinglobal = self.idx_nextglobal_to_nextcossinglobal
    idx_nextcossinglobal_to_nextcossin = self.idx_nextcossinglobal_to_nextcossin
    idx_nextrelative_to_next = self.idx_nextrelative_to_next
    idx_nextrelative_to_nextcossinrelative = self.idx_nextrelative_to_nextcossinrelative
    idx_nextcossinrelative_to_nextcossin = self.idx_nextcossinrelative_to_nextcossin
    
    for inextglobal in range(self.d_next_global):
      inext = idx_nextglobal_to_next[inextglobal]
      inextcossinglobal = idx_nextglobal_to_nextcossinglobal[inextglobal]
      inextcossin = idx_nextcossinglobal_to_nextcossin[inextcossinglobal]
      idx[inext] = inextcossin
      
    for inextrel in range(self.d_next_relative):
      inext = idx_nextrelative_to_next[inextrel]
      inextcossinrelative = idx_nextrelative_to_nextcossinrelative[inextrel]
      inextcossin = idx_nextcossinrelative_to_nextcossin[inextcossinrelative]
      idx[inext] = inextcossin
    return idx

  # which indices of nextcossin are discrete/continuous
  @property
  def idx_nextcossindiscrete_to_nextcossin(self):
    idx = []
    idx_next_to_nextcossin = self.idx_next_to_nextcossin
    for inext in self.idx_nextdiscrete_to_next:
      inextcossin = idx_next_to_nextcossin[inext]
      idx.append(inextcossin)
    return idx
  @property
  def idx_nextcossincontinuous_to_nextcossin(self):
    idx= []
    idx_next_to_nextcossin = self.idx_next_to_nextcossin
    for inext in self.idx_nextcontinuous_to_next:
      inextcossin = idx_next_to_nextcossin[inext]
      if type(inextcossin) is np.ndarray:
        idx.extend(inextcossin.tolist())
      else:
        idx.append(inextcossin)
    return idx

  @property
  def d_multi_relative(self):
    return self.d_next_cossin_relative*self.ntspred_relative

  @property
  def d_multi_global(self):
    return self.d_next_cossin_global*len(self.tspred_global)

  @property
  def d_multi(self):
    return self.d_multi_global + self.d_multi_relative

  # which multi correspond to nextcossin
  @property
  def idx_nextcossin_to_multi(self):
    assert (np.min(self.tspred_global) == 1)
    return self.feattpred_to_multi([(f,1) for f in range(self.d_next_cossin)])

  # look up table from multi index to (feat,tpred)
  # d_multi x 2 array
  @property
  def idx_multi_to_multifeattpred(self):
    return self.multi_to_feattpred(np.arange(self.d_multi))
  
  # look up table from (feat,tpred) to multi index
  # dict
  @property
  def idx_multifeattpred_to_multi(self):
    idx_multifeattpred_to_multi = {}
    for idx,ft in enumerate(self.idx_multi_to_multifeattpred):
      idx_multifeattpred_to_multi[tuple(ft.tolist())] = idx
    return idx_multifeattpred_to_multi

  # which indices of multi correspond to multi_relative and multi_global
  def get_multi_isrelative(self):
    idx_nextcossinrelative_to_nextcossin = self.idx_nextcossinrelative_to_nextcossin
    idx_multi_to_multifeattpred = self.idx_multi_to_multifeattpred
    isrelative = np.array([ft[0] in idx_nextcossinrelative_to_nextcossin for ft in idx_multi_to_multifeattpred])
    return isrelative
  @property
  def idx_multirelative_to_multi(self):
    isrelative = self.get_multi_isrelative()
    return np.nonzero(isrelative)[0]
  @property
  def idx_multiglobal_to_multi(self):
    isrelative = self.get_multi_isrelative()
    return np.nonzero(isrelative==False)[0]

  # which indices of multi correspond to multi_discrete, multi_continuous
  def get_multi_isdiscrete(self):
    idx_multi_to_multifeattpred = self.idx_multi_to_multifeattpred
    isdiscrete = (np.isin(idx_multi_to_multifeattpred[:,0],self.idx_nextcossindiscrete_to_nextcossin) & \
      (idx_multi_to_multifeattpred[:,1] == 1)) | \
      (np.isin(idx_multi_to_multifeattpred[:,0],self.idx_nextcossinglobal_to_nextcossin) & \
        np.isin(idx_multi_to_multifeattpred[:,1],self.discrete_tspred))
    return isdiscrete
  @property
  def idx_multidiscrete_to_multi(self):
    isdiscrete = self.get_multi_isdiscrete()
    return np.nonzero(isdiscrete)[0]  
  @property
  def idx_multicontinuous_to_multi(self):
    isdiscrete = self.get_multi_isdiscrete()
    return np.nonzero(isdiscrete==False)[0]
  @property
  def idx_multi_to_multidiscrete(self):
    isdiscrete = self.get_multi_isdiscrete()
    idx = np.zeros(self.d_multi,dtype=int)
    idx[:] = -1
    idx[isdiscrete] = np.arange(np.count_nonzero(isdiscrete))
    return idx
  @property
  def idx_multi_to_multicontinuous(self):
    iscontinuous = self.get_multi_isdiscrete() == False
    idx = np.zeros(self.d_multi,dtype=int)
    idx[:] = -1
    idx[iscontinuous] = np.arange(np.count_nonzero(iscontinuous))
    return idx
  
  @property
  def d_multidiscrete(self):
    return len(self.idx_multidiscrete_to_multi)
  @property
  def d_multicontinuous(self):
    return len(self.idx_multicontinuous_to_multi)

  def feattpred_to_multi(self,ftidx):
    idx = ravel_label_index(ftidx,ntspred_relative=self.ntspred_relative,
                            tspred_global=self.tspred_global,nrelrep=self.d_next_cossin_relative)
    return idx
  
  def multi_to_feattpred(self,idx):
    ftidx = unravel_label_index(idx,ntspred_relative=self.ntspred_relative,tspred_global=self.tspred_global,
                                nrelrep=self.d_next_cossin_relative)
    return ftidx

  def is_zscored(self):
    return self.zscore_params is not None
  
  def is_discretized(self):
    return self.discretize_params is not None
  
  def is_continuous(self):
    return 'continuous' in self.labels_raw
  
  def is_masked(self):
    return 'mask' in self.labels_raw
  
  def get_raw_labels(self,format='standard',ts=None,makecopy=True):
    labels_out = {}
    for kin in self.labels_raw.keys():
      if format == 'standard':
        kout = kin
      else:
        kout = self.label_keys[kin]
      if makecopy:
        labels_out[kout] = self.labels_raw[kin].copy()
      else:
        labels_out[kout] = self.labels_raw[kin]
      if ts is not None:
        labels_out[kout] = labels_out[kout][...,ts,:]
        
    labels_out['init'] = self.get_init_global(makecopy=makecopy)
    labels_out['scale'] = self.get_scale(makecopy=makecopy)
    labels_out['categories'] = self.get_categories(makecopy=makecopy)
        
    return labels_out
  
  def get_raw_labels_tensor_copy(self,**kwargs):
    raw_labels = self.get_raw_labels(makecopy=False,**kwargs)
    labels_out = {}
    for k,v in raw_labels.items():
      if type(v) is np.ndarray:
        labels_out[k] = torch.tensor(v)
    return labels_out
  
  def get_ntokens(self):
    return self.d_multidiscrete + int(self.is_continuous())
  
  def get_flatten_max_doutput(self):
    return np.max(self.d_multicontinuous,self.discretize_nbins)
  
  def get_train_labels(self,added_noise=None):

    # makes a copy
    raw_labels = self.get_raw_labels_tensor_copy()
    
    # to do: add noise
    assert added_noise is None, 'not implemented'
    
    train_labels = {}
    
    if self.is_discretized():
      train_labels['discrete'] = raw_labels['discrete'][...,self.starttoff:,:,:]
      train_labels['todiscretize'] = raw_labels['todiscretize'][...,self.starttoff:,:]
      train_labels['discrete_init'] = raw_labels['discrete'][...,:self.starttoff,:,:]
      train_labels['todiscretize_init'] = raw_labels['todiscretize'][...,:self.starttoff,:]
    else:
      train_labels['discrete'] = None
      train_labels['todiscretize'] = None
      train_labels['discrete_init'] = None
      train_labels['todiscretize_init'] = None

    train_labels['init_all'] = raw_labels['init']
    train_labels['init'] = raw_labels['init'][...,self.starttoff]
    train_labels['scale'] = raw_labels['scale']
    train_labels['categories'] = raw_labels['categories']
        
    if not self.flatten_labels:
      train_labels['continuous'] = raw_labels['continuous'][...,self.starttoff:,:]
      train_labels['continuous_init'] = raw_labels['continuous'][...,:self.starttoff,:]
      if 'mask' in raw_labels:
        train_labels['mask'] = raw_labels['mask'][...,self.starttoff:]
    else:
      contextl = self.ntimepoints
      dtype = raw_labels['continuous'].dtype
      ntokens =  self.get_ntokens()
      flatten_max_doutput = self.get_flatten_max_doutput()
      flatlabels = torch.zeros(self.pre_sz+(contextl,ntokens,flatten_max_doutput),dtype=dtype)
      for i in range(self.d_output_discrete):
        #inputnum = self.flatten_nobs_types+i
        flatlabels[...,i,:self.discretize_nbins] = raw_labels['discrete'][...,i,:]
        #newinput[:,inputnum,self.flatten_input_type_to_range[inputnum,0]:self.flatten_input_type_to_range[inputnum,1]] = raw_labels['labels_discrete'][:,i,:]
        # if mask is None:
        #   newmask[:,self.flatten_nobs_types+i] = True
        # else:
        #   newmask[:,self.flatten_nobs_types+i] = mask.clone()
      if self.continuous:
        #inputnum = -1
        flatlabels[...,-1,:self.d_multicontinuous] = raw_labels['continuous']
        #newinput[:,-1,self.flatten_input_type_to_range[inputnum,0]:self.flatten_input_type_to_range[inputnum,1]] = raw_labels['labels']
        # if mask is None:
        #   newmask[:,-1] = True
        # else:
        #   newmask[:,-1] = mask.clone()
      train_labels['continuous'] = flatlabels
      train_labels['continuous_stacked'] = raw_labels['continuous']
      train_labels['continuous_init'] = None
      
    return train_labels
  
  def get_mask(self,makecopy=True):
    if 'mask' not in self.labels_raw:
      return None
    if makecopy:
      return self.labels_raw['mask'].copy()
    else:
      return self.labels_raw['mask']
  
  def unzscore_multi(self,multi):
    if not self.is_zscored():
      return multi
    multi = unzscore(multi,self.zscore_params['mu_labels'],self.zscore_params['sig_labels'])
    return multi
  
  def zscore_multi(self,multi_unz):
    if not self.is_zscored():
      return multi_unz
    multi = zscore(multi_unz,self.zscore_params['mu_labels'],self.zscore_params['sig_labels'])
    return multi

  def labels_discrete_to_continuous(self,labels_discrete,epsilon=1e-3):
    assert self.is_discretized()
    sz = labels_discrete.shape
    nbins = sz[-1]
    nfeat = sz[-2]
    szrest = sz[:-2] 
    n = int(np.prod(np.array(szrest)))
    labels_discrete = labels_discrete.reshape((n,nfeat,nbins))
      
    # nfeat x nbins
    bin_centers = self.discretize_params['bin_medians']
    s = np.sum(labels_discrete,axis=-1)
    assert np.max(np.abs(1-s)) < epsilon, 'discrete labels do not sum to 1'
    continuous = np.sum(bin_centers[None,...]*labels_discrete,axis=-1) / s
    continuous = np.reshape(continuous,szrest+(nfeat,))
    return continuous

  def sample_discrete_labels(self,labels_discrete,nsamples=1):
    assert self.is_discretized()
    
    sz = labels_discrete.shape
    nbins = sz[-1]
    nfeat = sz[-2]
    szrest = sz[:-2] 
    n = int(np.prod(np.array(szrest)))
    labels_discrete = labels_discrete.reshape((n,nfeat,nbins))
    bin_samples = self.discretize_params['bin_samples']
    nsamples_per_bin = bin_samples.shape[0]
    continuous = np.zeros((nsamples,)+szrest+(nfeat,),dtype=labels_discrete.dtype)
    for f in range(nfeat):
      # to do make weighted_sample work with numpy directly
      binnum = weighted_sample(torch.tensor(labels_discrete[:,f,:]),nsamples=nsamples).numpy()
      sample = np.random.randint(low=0,high=nsamples_per_bin,size=(nsamples,n))
      curr = bin_samples[sample,f,binnum].reshape((nsamples,)+szrest)
      continuous[...,f] = curr
      
    return continuous
  
  def get_multi(self,use_todiscretize=False,nsamples=0,zscored=False,collapse_samples=False,ts=None):
    
    labels_raw = self.get_raw_labels(format='standard',ts=ts,makecopy=False)

    # to do: add flattening support here

    # allocate multi
    if ts is None:
      T = self.ntimepoints
    else:
      T = len(ts)
    multisz = self.pre_sz+(T,self.d_multi)
    if (nsamples > 1) or (nsamples == 1 and not collapse_samples):
      multisz = (nsamples,)+multisz
    multi = np.zeros(multisz,dtype=labels_raw['continuous'].dtype)
    multi[:] = np.nan
    
    if self.is_discretized():
      if use_todiscretize:
        assert 'todiscretize' in self.labels_raw
        # shape is pre_sz x T x d_multi_discrete
        labels_discrete = labels_raw['todiscretize']
      elif nsamples > 0:
        # shape is nsamples x pre_sz x T x d_multi_discrete
        labels_discrete = self.sample_discrete_labels(labels_raw['discrete'],nsamples)
        if nsamples == 1 and collapse_samples:
          labels_discrete = labels_discrete[0,...]
      else:
        labels_discrete = self.labels_discrete_to_continuous(labels_raw['discrete'])

      # store labels_discrete in multi
      multi[...,self.idx_multidiscrete_to_multi] = labels_discrete
        
    # get continuous
    multi[...,self.idx_multicontinuous_to_multi] = labels_raw['continuous']
    
    # unzscore
    if zscored == False and self.is_zscored():
      multi = self.unzscore_multi(multi)
      
    return multi
  
  def get_multi_discrete(self,makecopy=True,ts=None):
    if not self.is_discretized():
      nts = len_wrapper(ts,self.ntimepoints)
      return np.zeros((self.pre_sz+(nts,0,0)),dtype=self.dtype)
    labels_raw = self.get_raw_labels(format='standard',ts=ts,makecopy=makecopy)
    return labels_raw['discrete']
  
  def get_mask(self,makecopy=True,ts=None):
    if not self.is_masked():
      return None
    labels_raw = self.get_raw_labels(format='standard',ts=ts,makecopy=makecopy)
    return labels_raw['mask']
  
  def set_multi(self,multi,zscored=False,ts=None):
    
    multi = np.atleast_2d(multi)
    
    # zscore
    if self.is_zscored() and (zscored == False):
      multi = self.zscore_multi(multi)
      
    labels_raw = self.get_raw_labels(format='standard',makecopy=False)
    
    if ts is None:
      ts = slice(None)
    
    # set continuous
    labels_raw['continuous'] = multi[...,ts,self.idx_multicontinuous_to_multi]
    
    # set discrete
    if self.is_discretized():
      
      labels_raw['todiscretize'] = multi[...,ts,self.idx_multidiscrete_to_multi]
      labels_raw['discrete'] = discretize_labels(labels_raw['todiscretize'][...,ts,:],
                                                 self.discretize_params['bin_edges'],
                                                 soften_to_ends=True)
      
    return
  
  def multi_to_multiidct(self,multi):
    if not self.is_dct():
      return multi

    multi_idct = np.zeros(multi.shape,dtype=multi.dtype)
    idct_m = self.idct_m.T
    idx_nextcossinrelative_to_nextcossin = self.idx_nextcossinrelative_to_nextcossin
    idx_multi_to_multifeattpred = self.idx_multi_to_multifeattpred
    for irel in range(self.d_next_cossin_relative):
      i = idx_nextcossinrelative_to_nextcossin[irel]
      idxfeat = np.nonzero((idx_multi_to_multifeattpred[:,0] == i) & \
        (idx_multi_to_multifeattpred[:,1] > 1))[0]
      # features are in order
      assert np.all(idx_multi_to_multifeattpred[idxfeat,1] == np.arange(2,self.ntspred_relative+1))
      multi_dct = multi[...,idxfeat].reshape((-1,self.ntspred_relative-1))
      multi_idct[...,idxfeat] = (multi_dct @ idct_m).reshape((multi.shape[:-1])+(self.ntspred_relative-1,))
    return multi_idct
  
  def get_idx_mutli_to_futureglobal(self,tspred=None):
    idx_multi_to_multifeattpred = self.idx_multi_to_multifeattpred
    idx = np.isin(idx_multi_to_multifeattpred[:,0], self.idx_nextcossinglobal_to_nextcossin)
    if tspred is not None:
      idx = idx & (np.isin(idx_multi_to_multifeattpred[:,1],tspred))
    return idx
    
  def multi_to_futureglobal(self,multi,tspred=None):
    idx = self.get_idx_mutli_to_futureglobal(tspred)
    return multi[...,idx]

  def get_future_global(self,tspred=None,**kwargs):
    multi = self.get_multi(**kwargs)
    futureglobalvel = self.multi_to_futureglobal(multi,tspred=tspred)
    if tspred is None:
      ntspred = len(self.tspred_global)
    elif hasattr(tspred,'__len__'):
      ntspred = len(tspred)
    else:
      ntspred = 1
      tspred = [tspred,]
      
    futureglobalvel = futureglobalvel.reshape((futureglobalvel.shape[:-1])+(ntspred,self.d_next_cossin_global))
    
    return futureglobalvel

  def get_future_global_as_discrete(self,tspred=None,ts=None,**kwargs):
    # TODO: add some checks that global are discrete
    if not self.is_discretized():
      return None
    labels_raw = self.get_raw_labels(format='standard',ts=ts,makecopy=False)
    labels_discrete = np.zeros(self.pre_sz+(self.ntimepoints,self.d_multi,self.discretize_nbins),dtype=self.dtype)
    labels_discrete[:] = np.nan
    labels_discrete[...,self.idx_multidiscrete_to_multi,:] = labels_raw['discrete']
    idx = self.get_idx_mutli_to_futureglobal(tspred)
    labels_discrete = labels_discrete[...,idx,:]
    if tspred is None:
      ntspred = len(self.tspred_global)
    elif hasattr(tspred,'__len__'):
      ntspred = len(tspred)
    else:
      ntspred = 1
    
    labels_discrete = labels_discrete.reshape(self.pre_sz+(self.ntimepoints,ntspred,self.d_next_cossin_global,self.discretize_nbins))
    return labels_discrete
  
  def futureglobal_to_futureglobalpos(self,globalpos0,futureglobalvel,**kwargs):
    # futureglobalvel is szrest x T x ntspred x d_next_cossin_global

    szrest = futureglobalvel.shape[:-3]
    n = int(np.prod(szrest))
    T = futureglobalvel.shape[-3]
    ntspred = futureglobalvel.shape[-2]
    futureglobalvel = futureglobalvel.reshape((n,T,ntspred,self.d_next_global))
    globalpos0 = globalpos0[...,:T,:].reshape((n,T,self.d_next_global))
    xorigin0 = np.tile(globalpos0[...,None,:2],(1,1,ntspred,1))
    xtheta0 = np.tile(globalpos0[...,None,2],(1,1,ntspred))
    xoriginvelrel = futureglobalvel[...,[1,0]] # forward=y then sideways=x
    xoriginvel = mabe.rotate_2d_points(xoriginvelrel.reshape((n*T*ntspred,2)),-xtheta0.reshape(n*T*ntspred)).reshape((n,T,ntspred,2))
    xorigin = xorigin0 + xoriginvel
    xtheta = mabe.modrange(xtheta0 + futureglobalvel[...,2],-np.pi,np.pi)
    futureglobalpos = np.concatenate((xorigin,xtheta[...,None]),axis=-1)

    return futureglobalpos.reshape(szrest+(T,ntspred,self.d_next_global))
  
  def get_future_globalpos(self,tspred=None,**kwargs):
    globalpos0 = self.get_next_pose_global(**kwargs)
    futureglobal = self.get_future_global(tspred=tspred,**kwargs)
    futureglobalpos = self.futureglobal_to_futureglobalpos(globalpos0,futureglobal,**kwargs)
    return futureglobalpos
  
  def get_multi_idct(self,**kwargs):
    multi = self.get_multi(**kwargs)
    return self.multi_to_multiidct(multi)
  
  def multiidct_to_futurecossinrelative(self,multi_idct,tspred=None):
    if not self.is_dct():
      return np.zeros(self.pre_sz+(self.ntimepoints,0),dtype=multi_idct.dtype)
    if tspred is None:
      tspred = np.arange(2,self.ntspred_relative+1)
    elif not hasattr(tspred,'__len__'):
      tspred = [tspred,]
    ntspred = len(tspred)
    idx_multi_to_multifeattpred = self.idx_multi_to_multifeattpred
    idxfeat = np.nonzero(np.isin(idx_multi_to_multifeattpred[:,0],self.idx_nextcossinrelative_to_nextcossin) & \
        np.isin(idx_multi_to_multifeattpred[:,1], tspred))[0]
    return multi_idct[...,idxfeat].reshape((multi_idct.shape[:-1])+(ntspred,self.d_next_cossin_relative))

  def get_future_cossin_relative(self,tspred=None,**kwargs):

    multi_idct = self.get_multi_idct(**kwargs)
    futurerelcs = self.multiidct_to_futurecossinrelative(multi_idct,tspred=tspred)
    return futurerelcs

  def get_future_relative(self,tspred=None,**kwargs):
    futurerelcs = self.get_future_cossin_relative(tspred=tspred,**kwargs)
    futurerel = np.moveaxis(self.nextcossinrelative_to_nextrelative(np.moveaxis(futurerelcs,-2,0)),0,-2)
    return futurerel
  
  def get_future_relative_pose(self,tspred=None,**kwargs):
    futurerel = self.get_future_relative(tspred=tspred,**kwargs)
    if not self.is_velocity:
      return futurerel
    relpose0 = self.get_next_pose_relative(**kwargs)
    return futurerel + relpose0[...,:-1,None,:]
        
  def multi_to_nextcossin(self,multi):
    next_cossin = multi[...,self.idx_nextcossin_to_multi]
    return next_cossin
  
  def get_nextcossin(self,**kwargs):
    # note that multi_idct is ignored since we don't use the dct representation for the next frame
    multi = self.get_multi(**kwargs)
    return self.multi_to_nextcossin(multi)
  
  def set_nextcossin(self,nextcossin,**kwargs):
    nextcossin = np.atleast_2d(nextcossin)
    multi = self.get_multi(**kwargs)
    multi[...,self.idx_nextcossin_to_multi] = nextcossin
    self.set_multi(multi,**kwargs)
  
  def nextcossinglobal_to_nextglobal(self,next_cossinglobal):
    return next_cossinglobal
  
  def nextcossinrelative_to_nextrelative(self,next_cossin_relative):
    szrest = next_cossin_relative.shape[:-2]
    T = next_cossin_relative.shape[-2]
    n = int(np.prod(szrest))
    next_cossin_relative = next_cossin_relative.reshape((n,T,self.d_next_cossin_relative))
    next_relative = np.zeros((n,T,self.d_next_relative),dtype=next_cossin_relative.dtype)
    idx_nextrelative_to_nextcossinrelative = self.idx_nextrelative_to_nextcossinrelative
    for inext in range(self.d_next_relative):
      inextcossin = idx_nextrelative_to_nextcossinrelative[inext]
      if type(inextcossin) is np.ndarray:
        next_relative[...,inext] = np.arctan2(next_cossin_relative[...,inextcossin[1]],
                                              next_cossin_relative[...,inextcossin[0]])
      else:
        next_relative[...,inext] = next_cossin_relative[...,inextcossin]
    next_relative = next_relative.reshape(szrest+(T,self.d_next_relative))
    return next_relative
  
  def nextcossin_to_next(self,next_cossin):
    next = np.zeros(next_cossin.shape[:-1]+(self.d_next,),dtype=next_cossin.dtype)
    next[...,self.idx_nextglobal_to_next] = \
      self.nextcossinglobal_to_nextglobal(next_cossin[...,self.idx_nextcossinglobal_to_nextcossin])
    next[...,self.idx_nextrelative_to_next] = \
      self.nextcossinrelative_to_nextrelative(next_cossin[...,self.idx_nextcossinrelative_to_nextcossin])
    return next
  
  def next_to_nextcossin(self,next):
    szrest = next.shape[:-1]
    n = np.prod(szrest)
    next_cossin = np.zeros((n,self.d_next_cossin),dtype=next.dtype)
    idx_next_to_nextcossin = self.idx_next_to_nextcossin
    for inext in range(self.d_next):
      inextcossin = idx_next_to_nextcossin[inext]
      if type(inextcossin) is np.ndarray:
        next_cossin[...,inextcossin[0]] = np.cos(next[...,inext])
        next_cossin[...,inextcossin[1]] = np.sin(next[...,inext])
      else:
        next_cossin[...,inextcossin] = next[...,inext]
    return next_cossin
    
  def next_to_input_labels(self,next):
    return self.next_to_nextcossin(next)
  
  def get_input_labels(self,**kwargs):
    return self.get_nextcossin(zscored=True,use_todiscretize=True,**kwargs)
  
  def get_next(self,**kwargs):
    next_cossin = self.get_nextcossin(**kwargs)
    return self.nextcossin_to_next(next_cossin)
  
  def set_next(self,next,**kwargs):
    nextcossin = self.next_to_nextcossin(next)
    self.set_nextcossin(nextcossin,**kwargs)
    
  def convert_idx_next_to_nextcossin(self,idx_next):

    if not hasattr(idx_next,'__len__'):
      idx_next = [idx_next,]
      
    idx_next_to_nextcossin = self.idx_next_to_nextcossin
    
    idx_next_cossin = []
    for i in idx_next:
      ic = idx_next_to_nextcossin[i]
      if type(ic) is np.ndarray:
        idx_next_cossin = idx_next_cossin + ic.tolist()
      else:
        idx_next_cossin.append(ic)

    return idx_next_cossin
  
  def convert_idx_nextcossin_to_multi(self,idx_nextcossin):
    idx_nextcossin_to_multi = self.idx_nextcossin_to_multi
    idx_multi = idx_nextcossin_to_multi[idx_nextcossin]
    return idx_multi
  
  def convert_idx_next_to_multi(self,idx_next):
    idx_next_cossin = self.convert_idx_next_to_nextcossin(idx_next)
    idx_multi = self.convert_idx_nextcossin_to_multi(idx_next_cossin)
    
    return idx_multi

  def convert_idx_next_to_multi_anyt(self,idx_next):
    idx_next_cossin = self.convert_idx_next_to_nextcossin(idx_next)
    idx_multi_to_multifeattpred = self.idx_multi_to_multifeattpred
    idx_multi_anyt = np.nonzero(np.isin(idx_multi_to_multifeattpred[:,0],idx_next_cossin))[0]
    ts = idx_multi_to_multifeattpred[idx_multi_anyt,1]
    return idx_multi_anyt,ts

  def globalvel_to_globalpos(self,globalvel,starttoff=0,globalpos0=None):
    
    n = globalvel.shape[0]
    T = globalvel.shape[1]
    
    globalpos0 = self.init_pose[...,self.idx_nextglobal_to_next,starttoff]
    xorigin0 = globalpos0[...,:2]
    xtheta0 = globalpos0[...,2]
    
    thetavel = globalvel[...,2]
    xtheta = np.cumsum(np.concatenate((xtheta0.reshape((n,1)),thetavel),axis=-1),axis=-1)
    
    xoriginvelrel = globalvel[...,[1,0]] # forward=y then sideways=x
    xoriginvel = mabe.rotate_2d_points(xoriginvelrel.reshape((n*T,2)),-xtheta[...,:-1].reshape(n*T)).reshape((n,T,2))
    xorigin = np.cumsum(np.concatenate((xorigin0.reshape(n,1,2),xoriginvel),axis=-2),axis=-2)
    
    globalpos = np.concatenate((xorigin,xtheta[...,None]),axis=-1)
    return globalpos
  
  def relrep_to_relpose(self,relrep,starttoff=0):

    n = relrep.shape[0]
    T = relrep.shape[1]

    relpose0 = self.init_pose[...,self.idx_nextrelative_to_next,starttoff]
    
    if self.is_velocity:
      relpose = np.cumsum(np.concatenate((relpose0.reshape((n,1,-1)),relrep),axis=-2),axis=-2)
    else:
      relpose = np.concatenate((relpose0.reshape((n,1,-1)),relrep),axis=-2)
      
    return relpose

  def next_to_nextpose(self,next):

    szrest = next.shape[:-2]
    n = int(np.prod(szrest))
    starttoff = 0
    T = next.shape[-2]
    next = next.reshape((n,T,self.d_next))

    globalvel = next[...,self.idx_nextglobal_to_next]
    globalpos = self.globalvel_to_globalpos(globalvel,starttoff=starttoff)
    
    relrep = next[...,self.idx_nextrelative_to_next]
    relpose = self.relrep_to_relpose(relrep,starttoff=starttoff)

    pose = np.concatenate((globalpos,relpose),axis=-1)
    pose[...,self.is_angle_next] = mabe.modrange(pose[...,self.is_angle_next],-np.pi,np.pi)
    
    pose = pose.reshape(szrest+(pose.shape[-2],self.d_next))
    
    return pose
  
  def nextpose_to_next(self,nextpose):
    
    szrest = nextpose.shape[:-2]
    n = int(np.prod(szrest))
    T = nextpose.shape[-2]
    nextpose = nextpose.reshape((n,T,self.d_next))
    init_pose = nextpose[...,0,:]
    if self.is_velocity:
      next = np.diff(nextpose,axis=1)
    else:
      idx_nextglobal_to_next = self.idx_nextglobal_to_next
      next = nextpose[...,1:,:].copy()
      next[...,idx_nextglobal_to_next] = np.diff(nextpose[...,idx_nextglobal_to_next],axis=1)
    next[...,self.is_angle_next] = mabe.modrange(next[...,self.is_angle_next],-np.pi,np.pi)
    next = next.reshape(szrest+(T-1,self.d_next))
    
    return next,init_pose
  
  def next_to_nextvelocity(self,next):
    
    if self.is_velocity:
      return next
    
    szrest = next.shape[:-2]
    n = int(np.prod(szrest))
    T = next.shape[-2]
    idx_nextrelative_to_next = self.idx_nextrelative_to_next
    velrel = np.zeros((n,T+1,self.d_next_relative),dtype=next.dtype)
    velrel[:,0,:] = self.init_pose[idx_nextrelative_to_next]
    velrel[:,1:,:] = next[...,idx_nextrelative_to_next].reshape((n,T,self.d_next_relative))
    velrel[:,:-1,:] = np.diff(velrel,axis=1)
    velrel = velrel[:,:-1,:]
    velrel = velrel.reshape(szrest+(T,self.d_next_relative))
    vel = next.copy()
    vel[...,idx_nextrelative_to_next] = velrel
    
    return vel

  def get_next_pose(self,**kwargs):
    
    next = self.get_next(**kwargs)
    # global will always be velocity, still need to do an integration
    next_pose = self.next_to_nextpose(next)
    return next_pose
  
  def next_to_nextrelative(self,next):
    next_relative = next[...,self.idx_nextrelative_to_next]
    return next_relative

  def next_to_nextglobal(self,next):
    next_global = next[...,self.idx_nextglobal_to_next]
    return next_global
  
  def get_next_pose_relative(self,**kwargs):
    nextpose = self.get_next_pose(**kwargs)
    nextpose_relative = self.next_to_nextrelative(nextpose)
    return nextpose_relative

  def get_next_pose_global(self,**kwargs):
    nextpose = self.get_next_pose(**kwargs)
    nextpose_global = self.next_to_nextglobal(nextpose)
    return nextpose_global
  
  def set_next_pose(self,nextpose):
    self.pre_sz = nextpose.shape[:-2]
    next,init_pose = self.nextpose_to_next(nextpose)
    self.set_init_pose(init_pose.T)
    self.set_next(next,zscored=False)

  def nextpose_to_nextkeypoints(self,pose):
    
    if self.scale.ndim == 1:
      nflies = 1
    else:
      nflies = int(np.prod(self.scale.shape[:-1]))
    
    # input to mabe.feat2kp is expected to be an np.ndarray with shape nfeatures x T x nflies
    if nflies == 1:
      szrest = pose.shape[:-1]
      n = int(np.prod(szrest))
      pose = pose.reshape((n,self.d_next)).T
      scale = self.scale
    else:
      szrest = pose.shape[:-2]
      T = pose.shape[-2]
      n = int(np.prod(szrest))
      assert n == nflies
      pose = pose.reshape((n,T,self.d_next)).transpose((1,2,0))
      scale = self.scale.reshape((nflies,-1)).T
    kp = mabe.feat2kp(pose,scale)
    if nflies == 1:
      kp = kp[...,0].transpose((2,0,1))
      kp = kp.reshape(szrest+kp.shape[-2:])
    else:
      # kp will be nkpts x 2 x T x nflies
      kp= kp.transpose((3,2,0,1))
      # kp is now nflies x T x nkpts x 2
      kp = kp.reshape(szrest+kp.shape[1:])
    return kp
    
  def get_next_keypoints(self,**kwargs):
      
      next_pose = self.get_next_pose(zscored=False,**kwargs)
      next_keypoints = self.nextpose_to_nextkeypoints(next_pose)
      return next_keypoints
  
  def discretize_multi(self,example):
    if not self.is_discretized():
      return
    assert example['continuous'].shape[-1] == self.d_multi
    discretize_idx = self.idx_multidiscrete_to_multi
    example['todiscretize'] = example['continuous'][...,discretize_idx].copy()
    example['discrete'] = discretize_labels(example['todiscretize'],self.discretize_params['bin_edges'],soften_to_ends=True)
    example['continuous'] = example['continuous'][...,self.idx_multicontinuous_to_multi]
    return
    
  def set_keypoints(self,Xkp,scale=None):

    if (scale is None) and (self.scale is not None):
      scale = self.scale
      
    # function for computing features
    example = compute_features(Xkp[...,None],scale_perfly=scale,outtype=np.float32,
                               simplify_out=self.simplify_out,
                               dct_m=self.dct_m,
                               tspred_global=self.tspred_global,
                               compute_pose_vel=self.is_velocity,
                               discreteidx=self.idx_nextdiscrete_to_next,
                               simplify_in='no_sensory')
    
    if self.is_zscored():
      self.zscore(example)

    if self.is_discretized():
      self.discretize(example)
    
    self.set_raw_example(example)
    return
    
  def get_next_velocity(self,**kwargs):
    
    next = self.get_next(**kwargs)

    # global will always be velocity
    if self.is_velocity:
      return next

    next_vel = self.next_to_nextvelocity(next)

    return next_vel
  
  def set_zscore_params(self,zscore_params):
    self.zscore_params = zscore_params
    return
  
  def add_next_noise(self,eta_next,zscored=False):
    next = self.get_next(zscored=zscored)
    next = next + eta_next
    self.set_next(next,zscored=zscored)
    
  def get_nextglobal_names(self):
    return ['forward','sideways','orientation']
    
  def get_nextrelative_names(self):
    idx_nextrelative_to_next = self.idx_nextrelative_to_next
    return [mabe.posenames[i] for i in idx_nextrelative_to_next]
    
  def get_next_names(self):
    next_names = [None,]*self.d_next
    next_names_global = self.get_nextglobal_names()
    next_names_relative = self.get_nextrelative_names()
    for i,inext in enumerate(self.idx_nextglobal_to_next):
      next_names[inext] = next_names_global[i]
    for i,inext in enumerate(self.idx_nextrelative_to_next):
      next_names[inext] = next_names_relative[i]
    return next_names
  
  def get_nextcossin_names(self):
    next_names = self.get_next_names()
    idx_next_to_nextcossin = self.idx_next_to_nextcossin
    next_names_cossin = [None,]*self.d_next_cossin
    for i,ics in enumerate(idx_next_to_nextcossin):
      if hasattr(ics,'__len__'):
        next_names_cossin[ics[0]] = next_names[i]+'_cos'
        next_names_cossin[ics[1]] = next_names[i]+'_sin'
      else:
        next_names_cossin[ics] = next_names[i]
    return next_names_cossin
    
  def get_multi_names(self):
    ft = self.idx_multi_to_multifeattpred
    ismulti = (np.max(self.tspred_global) > 1) or (self.ntspred_relative > 1)
    multi_names = [None,]*self.d_multi
    nextcs_names = self.get_nextcossin_names()
    for i in range(self.d_multi):
      if ismulti:
        multi_names[i] = nextcs_names[ft[i,0]] + '_' + str(ft[i,1])    
    return multi_names
  
  def select_featidx_plot(self,ntsplot=None,ntsplot_global=None,ntsplot_relative=None):

    idx_multi_to_multifeattpred = self.idx_multi_to_multifeattpred
    idx_multifeattpred_to_multi = self.idx_multifeattpred_to_multi
    ntspred_global = len(self.tspred_global)
    if ntsplot_global is None and ntsplot is not None:
      ntsplot_global = ntsplot
    if ntsplot_global is None or (ntsplot >= ntspred_global):
      idxglobal = self.idx_multiglobal_to_multi
      ftglobal = idx_multi_to_multifeattpred[idxglobal,:]
      ntsplot_global = ntspred_global
    else:
      d_next_global = self.d_next_global
      tidxplot_global = np.concatenate((np.zeros((d_next_global,1),dtype=int),
                                        np.round(np.linspace(1,ntspred_global-1,(ntsplot_global-1)*d_next_global)).astype(int).reshape(-1,d_next_global).T),axis=-1)
      ftglobal = []
      idxglobal = []
      for fi,f in enumerate(self.idx_nextcossinglobal_to_nextcossin):
        for ti in range(ntsplot_global):
          ftcurr = (f,self.tspred_global[tidxplot_global[fi,ti]])
          ftglobal.append(ftcurr)
          idxglobal.append(idx_multifeattpred_to_multi[ftcurr])
      ftglobal = np.array(ftglobal)
      idxglobal = np.array(idxglobal)

    ntspred_relative = self.ntspred_relative
    if ntsplot_relative is None and ntsplot is not None:
      ntsplot_relative = ntsplot
    if ntsplot_relative is None or (ntsplot_relative >= ntspred_relative):
      idxrelative = self.idx_multirelative_to_multi
      ftrelative = idx_multi_to_multifeattpred[idxrelative,:]
      ntsplot_relative = ntspred_relative
    elif ntsplot_relative == 0:
      idxrelative = np.zeros((0,),dtype=int)
      ftrelative = np.zeros((0,2),dtype=int)
    else:
      d_next_cossin_relative = self.d_next_cossin_relative
      tplot_relative = np.concatenate((np.ones((d_next_cossin_relative,1),dtype=int),
                                        np.round(np.linspace(2,ntspred_relative,(ntsplot_relative-1)*d_next_cossin_relative)).astype(int).reshape(-1,d_next_cossin_relative).T),axis=-1)
      ftrelative = []
      idxrelative = []
      for fi,f in enumerate(self.idx_nextcossinrelative_to_nextcossin):
        for ti in range(ntsplot_relative):
          ftcurr = (f,tplot_relative[fi,ti])
          ftrelative.append(ftcurr)
          idxrelative.append(idx_multifeattpred_to_multi[ftcurr])
      ftrelative = np.array(ftrelative)
      idxrelative = np.array(idxrelative)
    idx = np.concatenate((idxglobal,idxrelative),axis=0)
    ft = np.concatenate((ftglobal,ftrelative),axis=0)
    order = np.argsort(idx,axis=0)
    idx = idx[order]
    ft = ft[order]

    return idx,ft
        
def dict_convert_torch_to_numpy(d):
  for k,v in d.items():
    if type(v) is torch.Tensor:
      d[k] = v.numpy()
    elif type(v) is dict:
      d[k] = dict_convert_torch_to_numpy(v)
  return d

class FlyExample:
  def __init__(self,example_in=None,dataset=None,Xkp=None,flynum=None,scale=None,metadata=None,
               dozscore=False,dodiscretize=False,**kwargs):

    self.set_params(kwargs)
    if dataset is not None:
      self.set_params(self.get_params_from_dataset(dataset),override=False)
    default_params = FlyExample.get_default_params()
    self.set_params(default_params,override=False)
    if self.dct_m is not None and self.idct_m is None:
      self.idct_m = np.linalg.inv(self.dct_m)

    # copy the dicts
    if example_in is not None:
      example_in = {k: v for k,v in example_in.items()}
      if 'metadata' in example_in:
        example_in['metadata'] = {k: copy.deepcopy(v) for k,v in example_in['metadata'].items()}
    elif Xkp is not None:
      example_in = self.compute_features(Xkp,flynum,scale,metadata)
      dozscore = True
      dodiscretize = True

    is_train_example = (example_in is not None) and ('input' in example_in) \
      and (type(example_in['input']) is torch.Tensor)
      
    if is_train_example:
      example_in = dict_convert_torch_to_numpy(example_in)
      # if we offset the example, adjust back
      if 't0' in example_in['metadata']:
        if 'labels_init' in example_in:
          starttoff = example_in['labels_init'].shape[-2]
        elif 'continuous_init' in example_in:
          starttoff = example_in['continuous_init'].shape[-2]
        else:
          starttoff = 0
        example_in['metadata']['t0'] -= starttoff
        example_in['metadata']['frame0'] -= starttoff
      
    self.labels = PoseLabels(example_in,dozscore=dozscore,dodiscretize=dodiscretize,**self.get_poselabel_params())

    if is_train_example and self.do_input_labels:
      self.remove_labels_from_input(example_in)

    self.inputs = ObservationInputs(example_in,dozscore=dozscore,**self.get_observationinputs_params())
    init_pose = self.inputs.get_init_next()
    
    self.labels.set_init_pose(init_pose)

    self.set_zscore_params(self.zscore_params)
    
    self.pre_sz = self.labels.pre_sz
    
    if (example_in is not None) and ('metadata' in example_in):
      self.metadata = example_in['metadata']
    
    return
  
  def compute_features(self,Xkp,flynum=0,scale=None,metadata=None):
    
    example = compute_features(Xkp,flynum=flynum,scale_perfly=scale,outtype=np.float32,
                               simplify_in=self.simplify_in,
                               simplify_out=self.simplify_out,
                               dct_m=self.dct_m,
                               tspred_global=self.tspred_global,
                               compute_pose_vel=self.is_velocity,
                               discreteidx=self.discreteidx)
        
    return example
  
  def copy(self):
    return self.copy_subindex()
  
  def copy_subindex(self,idx_pre=None,ts=None):
    
    example = self.get_raw_example(makecopy=True)
    
    if idx_pre is not None:
      ks = ['continuous', 'discrete', 'todiscretize', 'input','init','scale','categories']
      for k in ks:
        example[k] = example[k][idx_pre]
      for k in example['metadata'].keys():
        example['metadata'][k] = example['metadata'][k][idx_pre]
        
    if ts is not None:
      # hasn't been tested yet...
      ks = ['continuous', 'discrete', 'todiscretize', 'input']
      cattextra = example['categories'].shape[-1] - example['continuous'].shape[-2]
      if hasattr(ts,'__len__'):
        assert np.all(np.diff(ts) == 1), 'ts must be consecutive'
        toff = ts[0]
        example['categories'] = example['categories'][...,ts[0]:ts[-1]+cattextra,:]
      else:
        toff = ts
        example['categories'] = example['categories'][...,ts:ts+cattextra,:]
      for k in ks:
        if k == 'discrete':
          example[k] = example[k][...,ts,:,:]
        else:
          example[k] = example[k][...,ts,:]
      example['metadata']['t0'] += toff
      
    new = FlyExample(example_in=example,**self.get_params())
    return new

  def remove_labels_from_input(self,example_in):
    if not self.do_input_labels:
      return

    d_labels = self.labels.get_d_labels_input()
    example_in['input'] = example_in['input'][...,d_labels:]
      
  def set_params(self,params,override=True):
    synonyms = {'compute_pose_vel': 'is_velocity'}
    for k,v in params.items():
      if k in synonyms:
        k = synonyms[k]
      if override or (not hasattr(self,k)) or (getattr(self,k) is None):
        setattr(self,k,v)

  @staticmethod
  def get_default_params():
          
    params = {
      'zscore_params': None,
      'do_input_labels': True,
      'starttoff': 1,
      'flatten_labels': False,
      'flatten_obs': False,
      'discreteidx': [],
      'tspred_global': [1,],
      'discrete_tspred': [1,],
      'ntspred_relative': 1,
      'discretize_params': None,
      'is_velocity': False,
      'simplify_out': None,
      'simplify_in': None,
      'flatten_obs_idx': None,
      'dct_m': None,
      'idct_m': None,
    }
    return params
    
  def get_params(self):
    default_params = FlyExample.get_default_params()
    params = {k:getattr(self,k) for k in default_params.keys()}
    return params
  
  @staticmethod
  def get_params_from_dataset(dataset):
    params = {
      'zscore_params': dataset.get_zscore_params(),
      'do_input_labels': dataset.input_labels,
      'starttoff': dataset.get_start_toff(),
      'flatten_labels': dataset.flatten_labels,
      'flatten_obs': dataset.flatten_obs,
      'discreteidx': dataset.discretefeat,
      'tspred_global': dataset.tspred_global,
      'discrete_tspred': dataset.discrete_tspred,
      'ntspred_relative': dataset.ntspred_relative,
      'discretize_params': dataset.get_discretize_params(),
      'is_velocity': dataset.compute_pose_vel,
      'simplify_out': dataset.simplify_out,
      'simplify_in': dataset.simplify_in,
      'flatten_obs_idx': dataset.flatten_obs_idx,
      'dct_m': dataset.dct_m,
      'idct_m': dataset.idct_m,
    }
    return params
  
  def get_poselabel_params(self):
    params = self.get_params()
    params = PoseLabels.flyexample_to_poselabels_params(params)

    return params
  
  def get_observationinputs_params(self):
    params = self.get_params()
    params = ObservationInputs.flyexample_to_observationinput_params(params)
    return params
  
  @property
  def ntimepoints(self):
    # number of time points
    return self.labels.ntimepoints  
  
  @property
  def szrest(self):
    return self.labels.szrest
  
  def get_labels(self):
    return self.labels
  
  def get_inputs(self):
    return self.inputs
  
  def get_metadata(self,makecopy=True):
    if makecopy:
      return copy.deepcopy(self.metadata)
    else:
      return self.metadata
  
  def get_raw_example(self,format='standard',makecopy=True):
    example = self.labels.get_raw_labels(format=format,makecopy=makecopy)
    example['input'] = self.inputs.get_raw_inputs(makecopy=makecopy)
    example['metadata'] = self.get_metadata(makecopy=makecopy)
    return example
  
  def get_input_labels(self):
    if self.do_input_labels == False:
      return None
    else:
      return self.labels.get_input_labels()

  def get_train_example(self,do_add_noise=False):

    # to do: add noise
    metadata = self.get_train_metadata()
    input_labels = self.get_input_labels()

    train_inputs = self.inputs.get_train_inputs(input_labels=input_labels,
                                                    labels=self.labels,
                                                    do_add_noise=do_add_noise)
    train_labels = self.labels.get_train_labels(added_noise=train_inputs['eta'])
    
    flatten = self.flatten_labels or self.flatten_obs
    assert flatten == False, 'flatten not implemented'
        
    res = {'input': train_inputs['input'], 'labels': train_labels['continuous'], 
           'labels_discrete': train_labels['discrete'],
           'labels_todiscretize': train_labels['todiscretize'],
           'init': train_labels['init'], 'scale': train_labels['scale'], 
           'categories': train_labels['categories'],
           'metadata': metadata,
           'input_init': train_inputs['input_init'],
           'labels_init': train_labels['continuous_init'],
           'labels_discrete_init': train_labels['discrete_init'],
           'labels_todiscretize_init': train_labels['todiscretize_init'],
           'init_all': train_labels['init_all'],}

    return res
  
  def get_train_metadata(self):
    starttoff = self.starttoff
    metadata = copy.deepcopy(self.get_metadata())
    metadata['t0'] += starttoff
    metadata['frame0'] += starttoff    
    return metadata

  @staticmethod
  def split_zscore_params(zscore_params):
    if zscore_params is not None:
      zscore_params_input = {'mu_input':zscore_params['mu_input'],'sig_input':zscore_params['sig_input']}
      zscore_params_labels = {'mu_labels':zscore_params['mu_labels'],'sig_labels':zscore_params['sig_labels']}
    else:
      zscore_params_input = None
      zscore_params_labels = None
    return zscore_params_input,zscore_params_labels
  
  def set_zscore_params(self,zscore_params):
    zscore_params_input,zscore_params_labels = FlyExample.split_zscore_params(zscore_params)
    self.inputs.set_zscore_params(zscore_params_input)
    init_pose = self.inputs.get_init_next()
    self.labels.set_zscore_params(zscore_params_labels)
    self.labels.set_init_pose(init_pose)

class FlyMLMDataset(torch.utils.data.Dataset):
  def __init__(self,data,max_mask_length=None,pmask=None,masktype='block',
               simplify_out=None,simplify_in=None,pdropout_past=0.,maskflag=None,
               input_labels=True,
               dozscore=False,
               zscore_params={},
               discreteidx=None,
               discrete_tspred=[1,],
               discretize_nbins=50,
               discretize_epsilon=None,
               discretize_params={},
               flatten_labels=False,
               flatten_obs_idx=None,
               flatten_do_separate_inputs=False,
               input_noise_sigma=None,
               p_add_input_noise=0,
               dct_ms=None,
               tspred_global=[1,],
               compute_pose_vel=True):

    # copy dicts
    data = [example.copy() for example in data]
    self.dtype = data[0]['input'].dtype
    # number of outputs
    self.d_output = data[0]['labels'].shape[-1]
    self.d_output_continuous = self.d_output
    self.d_output_discrete = 0
    self.d_input = data[0]['input'].shape[-1]
    
    # number of inputs
    self.dfeat = data[0]['input'].shape[-1]
    
    self.max_mask_length = max_mask_length
    self.pmask = pmask
    self.masktype = masktype
    self.pdropout_past = pdropout_past
    self.simplify_out = simplify_out # modulation of task to make it easier
    self.simplify_in = simplify_in
    if maskflag is None:
      maskflag = (masktype is not None) or (pdropout_past>0.)
    self.maskflag = maskflag

    # TODO REMOVE THESE
    # features used for representing relative pose
    if compute_pose_vel:
      self.nrelrep = nrelative
      self.featrelative = featrelative.copy()
    else:
      self.relfeat_to_cossin_map,self.nrelrep = relfeatidx_to_cossinidx(discreteidx)
      self.featrelative = np.zeros(nglobal+self.nrelrep,dtype=bool)
      self.featrelative[nglobal:] = True
      
    self.discretefeat = discreteidx
    
    self.dct_m = None
    self.idct_m = None
    if dct_ms is not None:
      self.dct_m = dct_ms[0]
      self.idct_m = dct_ms[1]
    self.tspred_global = tspred_global
    
    # TODO REMOVE THESE
    # indices of labels corresponding to the next frame if multiple frames are predicted
    tnext = np.min(self.tspred_global)
    self.nextframeidx_global = self.ravel_label_index([(f,tnext) for f in featglobal])
    if self.simplify_out is None:
      self.nextframeidx_relative = self.ravel_label_index([(i,1) for i in np.nonzero(self.featrelative)[0]])
    else:
      self.nextframeidx_relative = np.array([])
    self.nextframeidx = np.r_[self.nextframeidx_global,self.nextframeidx_relative]
    if self.dct_m is not None:
      dct_tau = self.dct_m.shape[0]
      # not sure if t+1 should be t+2 here -- didn't add 1 when updating code to make t = 1 mean next frame for relative features
      self.idxdct_relative = np.stack([self.ravel_label_index([(i,t+1) for i in np.nonzero(self.featrelative)[0]]) for t in range(dct_tau)])
    self.d_output_nextframe = len(self.nextframeidx)
    
    # whether to predict relative pose velocities (true) or position (false)
    self.compute_pose_vel = compute_pose_vel
    
    if input_labels:
      assert(masktype is None)
      assert(pdropout_past == 0.)
      
    self.input_labels = input_labels
    # TODO REMOVE THESE
    if self.input_labels:
      self.d_input_labels = self.d_output_nextframe
    else:
      self.d_input_labels = 0
    
    # which outputs to discretize, which to keep continuous
    # TODO REMOVE THESE
    self.discreteidx = np.array([])
    
    self.discrete_tspred = np.array([1,])
    self.discretize = False

    # TODO REMOVE THESE
    self.continuous_idx = np.arange(self.d_output)
    
    self.discretize_nbins = None
    self.discretize_bin_samples = None
    self.discretize_bin_edges = None
    self.discretize_bin_means = None
    self.discretize_bin_medians = None
    
    self.mu_input = None
    self.sig_input = None
    self.mu_labels = None
    self.sig_labels = None
    
    self.dtype = np.float32
    
    # TODO REMOTE IDX
    self.flatten_labels = False
    self.flatten_obs_idx = None
    self.flatten_obs = False
    self.flatten_nobs_types = None
    self.flatten_nlabel_types = None
    self.flatten_dinput_pertype = None
    self.flatten_max_dinput = None
    self.flatten_max_doutput = None

    self.input_noise_sigma = input_noise_sigma
    self.p_add_input_noise = p_add_input_noise
    self.set_eval_mode()

    # apply all transforms to data
    if dozscore:
      print('Z-scoring data...')
      data = self.zscore(data,**zscore_params)
      print('Done.')

    if discreteidx is not None:
      print('Discretizing labels...')
      data = self.discretize_labels(data,discreteidx,discrete_tspred,nbins=discretize_nbins,bin_epsilon=discretize_epsilon,**discretize_params)
      print('Done.')
    
    self.set_flatten_params(flatten_labels=flatten_labels,flatten_obs_idx=flatten_obs_idx,flatten_do_separate_inputs=flatten_do_separate_inputs)

    # store examples in objects
    self.data = []
    for example_in in data:
      self.data.append(FlyExample(example_in,dataset=self))

    self.set_train_mode()

  @property
  def ntimepoints(self):
    # number of time points
    n = self.data[0].ntimepoints
    if self.input_labels and not (self.flatten_labels or self.flatten_obs) and not self.ismasked():
      n -= 1
    return n

  @property
  def ntokens_per_timepoint(self):
    if self.flatten_labels or self.flatten_obs:
      ntokens = self.flatten_nobs_types + self.flatten_nlabel_types
    else:
      ntokens = 1
    return ntokens

  @property
  def contextl(self):
    l = self.ntimepoints*self.ntokens_per_timepoint
    if (self.flatten_labels or self.flatten_obs) and not self.ismasked():
      l -= 1
    return l
  
  @property
  def flatten(self):
    return self.flatten_obs or self.flatten_labels
  
  @property
  def continuous(self):
    return (len(self.continuous_idx) > 0)
  
  @property
  def noutput_tokens_per_timepoint(self):
    if self.flatten_labels and self.discretize:
      return len(self.discreteidx) + int(self.continuous)
    else:
      return 1
    
  @property
  def dct_tau(self):
    if self.dct_m is None:
      return 0
    else:
      return self.dct_m.shape[0]
    
  @property
  def ntspred_relative(self):
    return self.dct_tau + 1
  
  @property
  def ntspred_global(self):
    return len(self.tspred_global)
          
  @property
  def ntspred_max(self):
    return np.maximum(self.ntspred_relative,self.ntspred_global)
          
  @property
  def is_zscored(self):
    return self.mu_input is not None
  
  def set_train_mode(self):
    self.do_add_noise = self.input_noise_sigma is not None and self.p_add_input_noise > 0

  def set_eval_mode(self):
    self.do_add_noise = False
          
  def set_flatten_params(self,flatten_labels=False,flatten_obs_idx=None,flatten_do_separate_inputs=False):
    
    # TODO REMOVE THESE
    self.flatten_labels = flatten_labels
    self.flatten_obs_idx = flatten_obs_idx
    if self.flatten_labels:
      if self.flatten_obs_idx is None:
        self.flatten_obs_idx = {'all': [0,self.dfeat]}
    self.flatten_obs = (self.flatten_obs_idx is not None) and (len(self.flatten_obs_idx) > 0)
    self.flatten_nobs_types = None
    self.flatten_nlabel_types = None
    self.flatten_dinput_pertype = None
    self.flatten_max_dinput = None
    self.flatten_max_doutput = None
    self.flatten_do_separate_inputs = flatten_do_separate_inputs
    
    if self.flatten_obs:
      self.flatten_nobs_types = len(self.flatten_obs_idx)
    else:
      self.flatten_nobs_types = 1
    if self.flatten_labels:
      self.flatten_nlabel_types = self.d_output_discrete
      if self.d_output_continuous > 0:
        self.flatten_nlabel_types += 1
    else:
      self.flatten_nlabel_types = 1
      
    if self.flatten:
      assert self.input_labels
      self.flatten_dinput_pertype = np.zeros(self.flatten_nobs_types+self.flatten_nlabel_types,dtype=int)
      for i,v in enumerate(self.flatten_obs_idx.values()):
        self.flatten_dinput_pertype[i] = v[1]-v[0]
      if self.flatten_labels and self.discretize:
        self.flatten_dinput_pertype[self.flatten_nobs_types:] = self.discretize_nbins
      if self.d_output_continuous > 0:
        self.flatten_dinput_pertype[-1] = self.d_output_continuous
      self.flatten_max_dinput = np.max(self.flatten_dinput_pertype)
      if self.flatten_do_separate_inputs:
        self.flatten_dinput = np.sum(self.flatten_dinput_pertype)        
      else:
        self.flatten_dinput = self.flatten_max_dinput

      self.flatten_input_type_to_range = np.zeros((self.flatten_dinput_pertype.size,2),dtype=int)
      
      if self.discretize:
        self.flatten_max_doutput = np.maximum(self.discretize_nbins,self.d_output_continuous)
      else:
        self.flatten_max_doutput = self.d_output_continuous

      if self.flatten_do_separate_inputs:
        cs = np.cumsum(self.flatten_dinput_pertype)
        self.flatten_input_type_to_range[1:,0] = cs[:-1]
        self.flatten_input_type_to_range[:,1] = cs
      else:
        self.flatten_input_type_to_range[:,1] = self.flatten_dinput_pertype

      
      # label tokens should be:
      # observations (flatten_nobs_types)
      # discrete outputs (d_output_discrete)
      # continuous outputs (<=1)
      self.idx_output_token_discrete = torch.arange(self.flatten_nobs_types,self.flatten_nobs_types+self.d_output_discrete,dtype=int)
      if self.d_output_continuous > 0:
        self.idx_output_token_continuous = torch.tensor([self.ntokens_per_timepoint-1,])
      else:
        self.idx_output_token_continuous = torch.tensor([])
        
    return
  
  # TODO REMOVE THESE
  def ravel_label_index(self,ftidx):
    
    idx = ravel_label_index(ftidx,dct_m=self.dct_m,tspred_global=self.tspred_global,nrelrep=self.nrelrep)
    return idx
  
  # TODO REMOVE THESE
  def unravel_label_index(self,idx):
      
    ftidx = unravel_label_index(idx,dct_m=self.dct_m,tspred_global=self.tspred_global,nrelrep=self.nrelrep)
    return ftidx
    
  def discretize_labels(self,data,discreteidx,discrete_tspred,nbins=50,
                        bin_edges=None,bin_samples=None,bin_epsilon=None,
                        bin_means=None,bin_medians=None,**kwargs):    
    """
    discretize_labels(self,discreteidx,discrete_tspred,nbins=50,bin_edges=None,bin_samples=None,bin_epsilon=None,**kwargs)
    For each feature in discreteidx, discretize the labels into nbins bins. For each example in the data,
    labels_discrete is an ndarray of shape T x len(discreteidx) x nbins, where T is the number of time points, and 
    indicates whether the label is in each bin, with soft-binning. 
    labels_todiscretize is an ndarray of shape T x len(discreteidx) with the original continuous labels.
    labels gets replaced with an ndarray of shape T x len(continuous_idx) with the continuous labels.
    discretize_bin_edges is an ndarray of shape len(discreteidx) x (nbins+1) with the bin edges for each discrete feature.
    discretize_bin_samples is an ndarray of shape nsamples x len(discreteidx) x nbins with samples from each bin
    """
        
    if not isinstance(discreteidx,np.ndarray):
      discreteidx = np.array(discreteidx)
    if not isinstance(discrete_tspred,np.ndarray):
      discrete_tspred = np.array(discrete_tspred)
      
    self.discrete_tspred = discrete_tspred

    bin_epsilon_feat = np.array(bin_epsilon)
    assert len(bin_epsilon_feat) <= len(discreteidx)
    if len(bin_epsilon_feat) < len(discreteidx):
      bin_epsilon_feat = np.concatenate((bin_epsilon_feat,np.zeros(len(discreteidx)-len(bin_epsilon_feat))))

    # translate to multi representation
    dummyexample = FlyExample(dataset=self)
    discreteidx_next = discreteidx
    bin_epsilon = np.zeros(dummyexample.labels.d_multi)
    bin_epsilon[:] = np.nan
    for i,i_next in enumerate(discreteidx_next):
      idx_multi_curr,_ = dummyexample.labels.convert_idx_next_to_multi_anyt(i_next)
      idx_multi_curr = idx_multi_curr[np.isin(idx_multi_curr,dummyexample.labels.idx_multidiscrete_to_multi)]
      bin_epsilon[idx_multi_curr] = bin_epsilon_feat[i]

    self.discreteidx = np.nonzero(np.isnan(bin_epsilon) == False)[0]
    self.bin_epsilon = bin_epsilon[self.discreteidx]
    
    self.discretize_nbins = nbins
    self.continuous_idx = np.ones(self.d_output,dtype=bool)
    self.continuous_idx[self.discreteidx] = False
    self.continuous_idx = np.nonzero(self.continuous_idx)[0]
    self.d_output_continuous = len(self.continuous_idx)
    self.d_output_discrete = len(self.discreteidx)

    assert((bin_edges is None) == (bin_samples is None))

    if bin_edges is None:
      if self.sig_labels is not None:
        bin_epsilon = np.array(self.bin_epsilon) / self.sig_labels[self.discreteidx]
      self.discretize_bin_edges,self.discretize_bin_samples,self.discretize_bin_means,self.discretize_bin_medians = \
        fit_discretize_labels(data,self.discreteidx,nbins=nbins,bin_epsilon=bin_epsilon,**kwargs)
    else:
      self.discretize_bin_samples = bin_samples
      self.discretize_bin_edges = bin_edges
      self.discretize_bin_means = bin_means
      self.discretize_bin_medians = bin_medians
      assert nbins == bin_edges.shape[-1]-1
        
    for example in data:
      example['labels_todiscretize'] = example['labels'][:,self.discreteidx]
      example['labels_discrete'] = discretize_labels(example['labels_todiscretize'],self.discretize_bin_edges,soften_to_ends=True)
      example['labels'] = example['labels'][:,self.continuous_idx]
    
    self.discretize = True    
    self.discretize_fun = lambda x: discretize_labels(x,self.discretize_bin_edges,soften_to_ends=True)
    
    return data
  
  def get_discretize_params(self):

    discretize_params = {
      'bin_edges': self.discretize_bin_edges,
      'bin_samples': self.discretize_bin_samples,
      'bin_means': self.discretize_bin_means,
      'bin_medians': self.discretize_bin_medians,
    }
    return discretize_params
    
  def get_bin_edges(self,zscored=False):

    if self.discretize == False:
      return
    
    if zscored or (self.mu_labels is None):
      bin_edges = self.discretize_bin_edges
    else:
      bin_edges = self.unzscore_labels(self.discretize_bin_edges.T,self.discreteidx).T
    
    return bin_edges
  
  def get_bin_samples(self,zscored=False):
    if self.discretize == False:
      return
    
    if zscored or (self.mu_labels is None):
      bin_samples = self.discretize_bin_samples
    else:
      sz = self.discretize_bin_samples.shape
      bin_samples = self.discretize_bin_samples.transpose(0,2,1).reshape((sz[0]*sz[2],sz[1]))
      bin_samples = self.unzscore_labels(bin_samples,self.discreteidx)
      bin_samples = bin_samples.reshape((sz[0],sz[2],sz[1])).transpose(0,2,1)
    
    return bin_samples
    
  def remove_labels_from_input(self,input):
    if self.hasmaskflag():
      return input[...,self.d_input_labels:-1]
    else:
      return input[...,self.d_input_labels:]
    
  def metadata_to_index(self,flynum,t0):
    starttoff = self.get_start_toff()
    for i,d in enumerate(self.data):
      if (d.metadata['t0'] == t0-starttoff) and (d.metadata['flynum'] == flynum):
        return i
    return None
    
  def hasmaskflag(self):
    return self.ismasked() or self.maskflag or self.pdropout_past > 0
    
  def ismasked(self):
    """Whether this object is a dataset for a masked language model, ow a causal model.
    v = self.ismasked()

    Returns:
        bool: Whether data are masked. 
    """
    return self.masktype is not None
    
  def zscore(self,data,mu_input=None,sig_input=None,mu_labels=None,sig_labels=None):
    """
    self.zscore(mu_input=None,sig_input=None,mu_labels=None,sig_labels=None)
    zscore the data. input and labels are z-scored for each example in data
    and converted to float32. They are stored in place in the dict for each example
    in the dataset. If mean and standard deviation statistics are input, then
    these statistics are used for z-scoring. Otherwise, means and standard deviations
    are computed from this data. 

    Args:
        mu_input (ndarray, dfeat, optional): Pre-computed mean for z-scoring input. 
        If None, mu_input is computed as the mean of all the inputs in data. 
        Defaults to None.
        sig_input (ndarray, dfeat, optional): Pre-computed standard deviation for 
        z-scoring input. If mu_input is None, sig_input is computed as the std of all 
        the inputs in data. Defaults to None. Do not set this to None if mu_input 
        is not None. 
        mu_labels (ndarray, d_output_continuous, optional): Pre-computed mean for z-scoring labels. 
        If None, mu_labels is computed as the mean of all the labels in data. 
        Defaults to None.
        sig_labels (ndarray, dfeat, optional): Pre-computed standard deviation for 
        z-scoring labels. If mu_labels is None, sig_labels is computed as the standard 
        deviation of all the labels in data. Defaults to None. Do not set this 
        to None if mu_labels is not None. 
        
    No value returned. 
    """
    
    # must happen before discretizing
    assert self.discretize == False, 'z-scoring should happen before discretizing'
    
    def zscore_helper(f):
      mu = 0.
      sig = 0.
      n = 0.
      for example in data:
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
    
    for example in data:
      example['input'] = self.zscore_input(example['input'])
      example['labels'] = self.zscore_labels(example['labels'])
      
    return data
      
  def get_poselabel_params(self):
    return self.data[0].labels.get_params()
  
  def get_flyexample_params(self):
    return self.data[0].get_params()
      
  def get_zscore_params(self):
    
    zscore_params = {
      'mu_input': self.mu_input.copy(),
      'sig_input': self.sig_input.copy(),
      'mu_labels': self.mu_labels.copy(),
      'sig_labels': self.sig_labels.copy(),
    }
    return zscore_params
      
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
  
  def zscore_nextframe_labels(self,rawlabels):
    if self.mu_labels is None:
      labels = rawlabels.copy()
    else:
      # if rawlabels.shape[-1] > self.d_output_continuous:
      #   labels = rawlabels.copy()
      #   labels[...,self.continuous_idx] = (rawlabels[...,self.continuous_idx]-self.mu_labels)/self.sig_labels
      # else:
      labels = (rawlabels-self.mu_labels[self.nextframeidx])/self.sig_labels[self.nextframeidx]
    return labels.astype(self.dtype)
    
  
  def zscore_labels(self,rawlabels):
    if self.mu_labels is None:
      labels = rawlabels.copy()
    else:
      # if rawlabels.shape[-1] > self.d_output_continuous:
      #   labels = rawlabels.copy()
      #   labels[...,self.continuous_idx] = (rawlabels[...,self.continuous_idx]-self.mu_labels)/self.sig_labels
      # else:
      labels = (rawlabels-self.mu_labels)/self.sig_labels
    return labels.astype(self.dtype)

  def unzscore_nextframe_labels(self,zlabels):
    if self.mu_labels is None:
      rawlabels = zlabels.copy()
    else:
      # if zlabels.shape[-1] > self.d_output_continuous:
      #   rawlabels = zlabels.copy()
      #   rawlabels[...,self.continuous_idx] = unzscore(zlabels[...,self.continuous_idx],self.mu_labels,self.sig_labels)
      # else:
      rawlabels = unzscore(zlabels,self.mu_labels[self.nextframeidx],self.sig_labels[self.nextframeidx])
    return rawlabels.astype(self.dtype)


  def unzscore_labels(self,zlabels,featidx=None):
    if self.mu_labels is None:
      rawlabels = zlabels.copy()
    else:
      # if zlabels.shape[-1] > self.d_output_continuous:
      #   rawlabels = zlabels.copy()
      #   rawlabels[...,self.continuous_idx] = unzscore(zlabels[...,self.continuous_idx],self.mu_labels,self.sig_labels)
      # else:
      if featidx is None:
        rawlabels = unzscore(zlabels,self.mu_labels,self.sig_labels)
      else:
        rawlabels = unzscore(zlabels,self.mu_labels[featidx],self.sig_labels[featidx])
    return rawlabels.astype(self.dtype)

  
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
  
  def get_input_shapes(self):
    idx,sz = get_sensory_feature_shapes(self.simplify_in)
    if self.input_labels:
      for k,v in idx.items():
        idx[k] = [x + self.d_input_labels for x in v]
      idx['labels'] = [0,self.d_input_labels]
      sz['labels'] = (self.d_input_labels,)
    return idx,sz

  def unpack_input(self,input,dim=-1):
    
    idx,sz = self.get_input_shapes()
    res = unpack_input(input,idx,sz,dim=dim)
    
    return res
    
  def get_start_toff(self):
    if self.ismasked() or (self.input_labels == False) or \
      self.flatten_labels or self.flatten_obs:
      starttoff = 0
    else:
      starttoff = 1
    return starttoff
    
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
        example['input'] is a tensor of shape contextl x (d_input_labels + dfeat + 1)
        where example['input'][t,:d_input_labels] is the motion from frame t to t+1 and
        example['input'][t,d_input_labels:-1] are the input features for frame t. 
        example['input'][t,-1] indicates whether the frame is masked or not. If this 
        frame is masked, then example['input'][t,:d_input_labels] will be set to 0. 
        example['labels'] is a tensor of shape contextl x d_output_continuous
        where example['labels'][t,:] is the continuous motion from frame t to t+1 and/or
        the pose at frame t+1
        example['labels_discrete'] is a tensor of shape contextl x d_output_discrete x 
        discretize_nbins, where example['labels_discrete'][t,i,:] is one-hot encoding of 
        discrete motion feature i from frame t to t+1 and/or pose at frame t+1. 
        example['init'] is a tensor of shape dglobal, corresponding to the global
        position in frame 0. 
        example['mask'] is a tensor of shape contextl indicating which frames are masked.
        
        For causal LMs:
        example['input'] is a tensor of shape (contextl-1) x (d_input_labels + dfeat).
        if input_labels == True, example['input'][t,:d_input_labels] is the motion from 
        frame t to t+1 and/or the pose at frame t+1,
        example['input'][t,d_input_labels:] are the input features for 
        frame t+1. 
        example['labels'] is a tensor of shape contextl x d_output
        where example['labels'][t,:] is the motion from frame t+1 to t+2 and/or the pose at
        frame t+2.
        example['init'] is a tensor of shape dglobal, corresponding to the global
        position in frame 1. 
        example['labels_discrete'] is a tensor of shape contextl x d_output_discrete x 
        discretize_nbins, where example['labels_discrete'][t,i,:] is one-hot encoding of 
        discrete motion feature i from frame t+1 to t+2 and/or pose at frame t+2.

        For all:
        example['scale'] are the scale features for this fly, used for converting from
        relative pose features to keypoints. 
        example['categories'] are the currently unused categories for this sequence.
        example['metadata'] is a dict of metadata about this sequence.
        
    """
    
    res = self.data[idx].get_train_example()
    res['input'],mask,dropout_mask = self.mask_input(res['input'])
    if mask is not None:
      res['mask'] = mask
    if dropout_mask is not None:
      res['dropout_mask'] = dropout_mask
      
    return res
    
    # TODO REMOVE THIS AFTER CHECKING FLATTENING
    datacurr = copy.deepcopy(self.data[idx])
    
    if self.input_labels:
      # should we use all future predictions, or just the next time point?
      input_labels = datacurr.labels.get_input_labels()
    else:
      input_labels = None
    
    # add_noise
    # to do: make this work with objects
    if self.do_add_noise:
      eta,datacurr = self.add_noise(datacurr,input_labels)
    if self.input_labels:
      input_labels = torch.tensor(input_labels)
    labels = datacurr.get_labels()
    
    # whether we start with predicting the 0th or the 1th frame in the input sequence
    starttoff = self.get_start_toff()

    init = torch.tensor(labels.get_init_pose(starttoff))
    scale = torch.tensor(labels.get_scale())
    categories = torch.tensor(labels.get_categories())
    metadata = datacurr.get_metadata(makecopy=True)
    metadata['t0'] += starttoff
    metadata['frame0'] += starttoff

    raw_labels = labels.get_raw_labels_tensor_copy(format='input')
    res = {'input': None, 'labels': None, 'labels_discrete': None,
           'labels_todiscretize': None,
           'init': init, 'scale': scale, 'categories': categories,
           'metadata': metadata}

    res['labels'] = raw_labels['labels'][starttoff:,:]
    if self.discretize:
      res['labels_discrete'] = raw_labels['labels_discrete'][starttoff:,:,:]
      res['labels_todiscretize'] = raw_labels['labels_todiscretize'][starttoff:,:]

    input = torch.tensor(datacurr.get_inputs().get_raw_inputs())
    nin = input.shape[-1]
    contextl = input.shape[0]
    input,mask,dropout_mask = self.mask_input(input)
    
    if self.flatten:
      ntypes = self.ntokens_per_timepoint
      #newl = contextl*ntypes
      newlabels = torch.zeros((contextl,ntypes,self.flatten_max_doutput),dtype=input.dtype)
      newinput = torch.zeros((contextl,ntypes,self.flatten_dinput),dtype=input.dtype)
      newmask = torch.zeros((contextl,ntypes),dtype=bool)
      #offidx = np.arange(contextl)*ntypes
      if self.flatten_obs:
        for i,v in enumerate(self.flatten_obs_idx.values()):
          newinput[:,i,self.flatten_input_type_to_range[i,0]:self.flatten_input_type_to_range[i,1]] = input[:,v[0]:v[1]]
          newmask[:,i] = False
      else:
        newinput[:,0,:self.flatten_dinput_pertype[0]] = input
      if self.discretize:
        if self.flatten_labels:
          for i in range(self.d_output_discrete):
            inputnum = self.flatten_nobs_types+i
            newlabels[:,inputnum,:self.discretize_nbins] = raw_labels['labels_discrete'][:,i,:]
            newinput[:,inputnum,self.flatten_input_type_to_range[inputnum,0]:self.flatten_input_type_to_range[inputnum,1]] = raw_labels['labels_discrete'][:,i,:]
            if mask is None:
              newmask[:,self.flatten_nobs_types+i] = True
            else:
              newmask[:,self.flatten_nobs_types+i] = mask.clone()
          if self.continuous:
            inputnum = -1
            newlabels[:,-1,:labels.shape[-1]] = raw_labels['labels']
            newinput[:,-1,self.flatten_input_type_to_range[inputnum,0]:self.flatten_input_type_to_range[inputnum,1]] = raw_labels['labels']
            if mask is None:
              newmask[:,-1] = True
            else:
              newmask[:,-1] = mask.clone()
        else:
          newinput[:,-1,:self.d_output] = raw_labels['labels']
      newlabels = newlabels.reshape((contextl*ntypes,self.flatten_max_doutput))
      newinput = newinput.reshape((contextl*ntypes,self.flatten_dinput))
      newmask = newmask.reshape(contextl*ntypes)
      if not self.ismasked():
        newlabels = newlabels[1:,:]
        newinput = newinput[:-1,:]
        newmask = newmask[1:]
        
      res['input'] = newinput
      res['input_stacked'] = input
      res['mask_flattened'] = newmask
      res['labels'] = newlabels
      res['labels_stacked'] = labels
    else:
      if self.input_labels:
        input = torch.cat((input_labels[:-starttoff,:],input[starttoff:,:]),dim=-1)
      res['input'] = input

    if mask is not None:
      res['mask'] = mask
    if dropout_mask is not None:
      res['dropout_mask'] = dropout_mask
    return res
    
  # TODO REMOVE THIS AFTER CHECKING ADDING NOISE
  
  def add_noise(self,example,input_labels):

    # add noise to the input movement and pose
    # desire is to do movement truemovement(t-1->t)
    # movement noisemovement(t-1->t) = truemovement(t-1->t) + eta(t) actually done
    # resulting in noisepose(t) = truepose(t) + eta(t)[featrelative]
    # output should then be fixmovement(t->t+1) = truemovement(t->t+1) - eta(t)
    # input pose: noise_input_pose(t) = truepose(t) + eta(t)[featrelative]
    # input movement: noise_input_movement(t-1->t) = truemovement(t-1->t) + eta(t)
    # output movement: noise_output_movement(t->t+1) = truemovement(t->t+1) - eta(t)
    # movement(t->t+1) = truemovement(t->t+1)

    example = copy.deepcopy(example)

    T = example.ntimesteps
    d_labels = example.labels.d_next
    
    # # divide sigma by standard deviation if zscored
    # if self.sig_labels is not None:
    #   input_noise_sigma = self.input_noise_sigma / self.sig_labels

    # additive noise
    eta = np.zeros((T,d_labels))
    do_add_noise = np.random.rand(T) <= self.p_add_input_noise
    eta[do_add_noise,:] = self.input_noise_sigma[None,:]*np.random.randn(np.count_nonzero(do_add_noise),self.d_output)

    # problem with multiplicative noise is that it is 0 when the movement is 0 -- there should always be some jitter
    # etamult = np.maximum(-self.max_input_noise,np.minimum(self.max_input_noise,self.input_noise_sigma[None,:]*np.random.randn(input.shape[0],self.d_output)))
    # if self.input_labels:
    #   eta = input_labels*etamult
    # else:
    #   labelsprev = torch.zeros((labels.shape[0],nfeatures),dtype=labels.dtype,device=labels.device)
    #   if self.continuous:
    #     labelsprev[1:,self.continuous_idx] = labels[:-1,:]
    #   if self.discretize:
    #     labelsprev[1:,self.discreteidx] = labels_todiscretize[:-1,:]
    #   eta = labelsprev*etamult

    # input pose
    eta_pose = example.labels.next_to_nextpose(eta)
    example.inputs.add_pose_noise(eta_pose,zscored=False)

    # input labels
    if self.input_labels:
      eta_input_labels = example.labels.next_to_input_labels(eta)
      input_labels += eta_input_labels

    # output labels
    example.labels.add_next_noise(-eta,zscored=False)

    return eta
  
  def __len__(self):
    return len(self.data)
  
  # TODO REMOVE ALL OF THESE
  def get_global_movement_idx(self):
    idxglobal = self.ravel_label_index(np.stack(np.meshgrid(featglobal,self.tspred_global),axis=-1))
    return idxglobal
  
  def get_global_movement(self,movement):
    idxglobal = self.get_global_movement_idx()
    movement_global = movement[...,idxglobal]
    return movement_global
  
  def set_global_movement(self,movement_global,movement):
    idxglobal = self.get_global_movement_idx()
    movement[...,idxglobal] = movement_global
    return movement
  
  def get_global_movement_discrete(self,movement_discrete):
    if not self.discretize:
      return None
    idxglobal = self.get_global_movement_idx()
    movement_global_discrete = np.zeros(movement_discrete.shape[:-2]+idxglobal.shape+movement_discrete.shape[-1:],dtype=self.dtype)
    movement_global_discrete[:] = np.nan
    for i in range(idxglobal.shape[0]):
      for j in range(idxglobal.shape[1]):
        idx = idxglobal[i,j]
        didx = np.nonzero(self.discreteidx==idx)[0]
        if len(didx) == 0:
          continue
        movement_global_discrete[...,i,j,:] = movement_discrete[...,didx[0],:]
    return movement_global_discrete

  def get_next_relative_movement(self,movement):
    movement_next_relative = movement[...,self.nextframeidx_relative]
    return movement_next_relative

  def get_relative_movement_dct(self,movements,iszscored=False):
    movements_dct = movements[...,self.idxdct_relative]
    if not iszscored and self.mu_labels is not None:
      movements_dct = unzscore(movements_dct,self.mu_labels[self.idxdct_relative],self.sig_labels[self.idxdct_relative])
    movements_relative = self.idct_m @ movements_dct
    return movements_relative
  
  def get_next_relative_movement_dct(self,movements,iszscored=True,dozscore=True):
    if self.simplify_out == 'global':
      return movements[...,[]]
    
    if type(movements) is np.ndarray:
      movements = torch.as_tensor(movements)

    movements_dct = movements[...,self.idxdct_relative]
    if not iszscored and self.mu_labels is not None:
      mu = torch.as_tensor(self.mu_labels[self.idxdct_relative]).to(dtype=movements.dtype,device=movements.device)
      sig = torch.as_tensor(self.sig_labels[self.idxdct_relative]).to(dtype=movements.dtype,device=movements.device)
      movements_dct = unzscore(movements_dct,mu,sig)
      
    idct_m0 = torch.as_tensor(self.idct_m[[0,],:]).to(dtype=movements.dtype,device=movements.device)
    dctfeat = movements[...,self.idxdct_relative]
    movements_next_relative = torch.matmult(idct_m0,dctfeat)
    
    if dozscore:
      movements_next_relative = zscore(movements_next_relative,self.mu_labels[self.nextframeidx_relative],self.sig_labels[self.nextframeidx_relative])
      
    return movements_next_relative
  
  def compare_dct_to_next_relative(self,movements):
    movements_next_relative_dct = self.get_next_relative_movement_dct(movements,iszscored=True,dozscore=True)
    movements_next_relative0 = movements[...,self.nextframeidx_relative]
    err = movements_next_relative_dct - movements_next_relative0
    return err
  
  def get_next_movements(self,movements=None,example=None,iszscored=False,use_dct=False,**kwargs):
    """
    get_next_movements(movements=None,example=None,iszscored=False,use_dct=False,**kwargs)
    extracts the next frame movements/pose from the input, ignoring predictions for frames further
    into the future. 
    Inputs:
      movements: ... x d_output ndarray of movements. Required if example is None. Default: None. 
      example: dict holding training/test example. Required if movements is None. Default: None.
      iszscored: whether movements are z-scored. Default: False.
      use_dct: whether to use DCT to extract relative pose features. Default: False.
      Extra args are fed into get_full_labels if movements is None
    Outputs:
      movements_next: ... x d_output ndarray of movements/pose for the next frame.
    """
    if movements is None:
      movements = self.get_full_labels(example=example,**kwargs)
      iszscored = True
      
    if torch.is_tensor(movements):
      movements = movements.numpy()
      
    if iszscored and self.mu_labels is not None:
      movements = unzscore(movements,self.mu_labels,self.sig_labels)
    
    movements_next_global = movements[...,self.nextframeidx_global]
    if self.simplify_out is None:
      if use_dct and self.dct_m is not None:
        dctfeat = movements[...,self.idxdct_relative]
        movements_next_relative = self.idct_m[[0,],:] @ dctfeat
      else:
        movements_next_relative = movements[...,self.nextframeidx_relative]
      movements_next = np.concatenate((movements_next_global,movements_next_relative),axis=-1)
    else:
      movements_next = movements_next_global
    return movements_next
  
  
  def get_init_pose(self,example=None,input0=None,global0=None,zscored=False):
    if example is not None:
      if input0 is None:
        input = self.get_full_inputs(example=example)
        input0 = input[...,0,:]
      if global0 is None:
        global0 = example['init']

    istorch = torch.is_tensor(input0)

    if (self.mu_input is not None) and (zscored==False):
      input0 = unzscore(input0,self.mu_input,self.sig_input)

    input0 = split_features(input0,simplify=self.simplify_in)
    relative0 = input0['pose']
    
    if istorch:
      pose0 = torch.zeros(nfeatures,dtype=relative0.dtype,device=relative0.device)
    else:
      pose0 = np.zeros(nfeatures,dtype=relative0.dtype)
    pose0[featglobal] = global0
    pose0[featrelative] = relative0
      
    return pose0
  
  def get_Xfeat(self,input0=None,global0=None,movements=None,example=None,use_dct=False,**kwargs):
    """
    Xfeat = self.get_Xfeat(input0,global0,movements)
    Xfeat = self.get_Xfeat(example=example)

    Unnormalizes initial input input0 and extracts relative pose features. Combines
    these with global0 to get the full set of pose features for initial frame 0. 
    Converts egocentric movements (forward, sideway) to global, and computes the
    full pose features for each frame based on the input movements. 

    Either input0, global0, and movements must be input OR 
    example must be input, and input0, global0, and movements are derived from there.

    Args:
        input0 (ndarray, d_input_labels+dfeat+hasmaskflag): network input for time point 0
        global0 (ndarray, 3): global position at time point 0
        movements (ndarray, T x d_output ): movements[t,:] is the movement from t to t+1

    Returns:
        Xfeat: (ndarray, T+1 x nfeatures): All pose features for frames 0 through T
    """
    
    if example is not None:
      if input0 is None:
        input = self.get_full_inputs(example=example)
        input0 = input[...,0,:]
      if global0 is None:
        global0 = example['init']
      if movements is None:
        movements = self.get_full_labels(example=example,**kwargs)

    szrest = movements.shape[:-1]
    n = np.prod(np.array(szrest))
    
    if torch.is_tensor(input0):
      input0 = input0.numpy()
    if torch.is_tensor(global0):
      global0 = global0.numpy()
    if torch.is_tensor(movements):
      movements = movements.numpy()
    
    if self.mu_input is not None:
      input0 = unzscore(input0,self.mu_input,self.sig_input)
      movements = unzscore(movements,self.mu_labels,self.sig_labels)

    # get movements/pose for next frame prediction
    movements_next = self.get_next_movements(movements=movements,iszscored=False,use_dct=use_dct)
      
    if not self.compute_pose_vel:
      movements_next = self.convert_cos_sin_to_angle(movements_next)
      
    input0 = split_features(input0,simplify=self.simplify_in)
    Xorigin0 = global0[...,:2]
    Xtheta0 = global0[...,2] 
    thetavel = movements_next[...,feattheta]
    
    Xtheta = np.cumsum(np.concatenate((Xtheta0[...,None],thetavel),axis=-1),axis=-1)
    Xoriginvelrel = movements_next[...,[featorigin[1],featorigin[0]]]
    Xoriginvel = mabe.rotate_2d_points(Xoriginvelrel.reshape((n,2)),-Xtheta[...,:-1].reshape(n)).reshape(szrest+(2,))
    Xorigin = np.cumsum(np.concatenate((Xorigin0[...,None,:],Xoriginvel),axis=-2),axis=-2)
    Xfeat = np.zeros(szrest[:-1]+(szrest[-1]+1,nfeatures),dtype=self.dtype)
    Xfeat[...,featorigin] = Xorigin
    Xfeat[...,feattheta] = Xtheta

    if self.simplify_out == 'global':
      Xfeat[...,featrelative] = np.tile(input0['pose'],szrest[:-1]+(szrest[-1]+1,1))
    else:
      Xfeatpose = np.concatenate((input0['pose'][...,None,:],movements_next[...,featrelative]),axis=-2)
      if self.compute_pose_vel:
        Xfeatpose = np.cumsum(Xfeatpose,axis=-2)
      Xfeat[...,featrelative] = Xfeatpose
    
    return Xfeat
  
  def get_Xkp(self,example,pred=None,**kwargs):
    """
    Xkp = self.get_Xkp(example,pred=None)

    Call get_Xfeat to get the full pose features based on the initial input and global
    position example['input'] and example['init'] and the per-frame motion in 
    pred (if not None) or example['labels'], example['labels_discrete']. Converts
    the full pose features to keypoint coordinates. 
    
    Args:
        scale (ndarray, dscale): scale parameters for this fly
        example (dict), output of __getitem__: example with fields input, init, labels, and
        scale. 
        pred (ndarray, T x d_output ): movements[t,:] is the movement from t to t+1

    Returns:
        Xkp: (ndarray, nkeypoints x 2 x T+1 x 1): Keypoint locations for frames 0 through T
    """
    
    scale = example['scale']
    if torch.is_tensor(scale):
      scale = scale.numpy()

    if pred is not None:
      movements = self.get_full_pred(pred,**kwargs)
    else:
      movements = None
    Xfeat = self.get_Xfeat(example=example,movements=movements,**kwargs)
    Xkp = self.feat2kp(Xfeat,scale)
    return Xkp

  
  def get_Xkp0(self,input0=None,global0=None,movements=None,scale=None,example=None):
    """
    Xkp = self.get_Xkp(input0,global0,movements)

    Call get_Xfeat to get the full pose features based on the initial input and global
    position input0 and global0 and the per-frame motion in movements. Converts
    the full pose features to keypoint coordinates. 
    
    Either input0, global0, movements, and scale must be input OR 
    example must be input, and input0, global0, movements, and scale are derived from there

    Args:
        input0 (ndarray, d_input_labels+dfeat+hasmaskflag): network input for time point 0
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
  
  def construct_input(self,obs,movement=None):    
    
    # to do: merge this code with getitem so that we don't have to duplicate
    dtype = obs.dtype
    
    if self.input_labels:
      assert (movement is not None)
    
    if self.flatten:
      xcurr = np.zeros((obs.shape[0],self.ntokens_per_timepoint,self.flatten_dinput),dtype=dtype)

      if self.flatten_obs:
        for i,v in enumerate(self.flatten_obs_idx.values()):
          xcurr[:,i,self.flatten_input_type_to_range[i,0]:self.flatten_input_type_to_range[i,1]] = obs[:,v[0]:v[1]]
      else:
        xcurr[:,0,:self.flatten_dinput_pertype[0]] = obs
  
      if self.input_labels:
         # movement not set for last time points, will be 0s
        if self.flatten_labels:
          for i in range(movement.shape[1]):
            if i < len(self.discreteidx):
              dmovement = self.discretize_nbins
            else:
              dmovement = len(self.continuous_idx)
            inputnum = self.flatten_nobs_types+i
            xcurr[:-1,inputnum,self.flatten_input_type_to_range[inputnum,0]:self.flatten_input_type_to_range[inputnum,1]] = movement[:,i,:dmovement]
        else:
          inputnum = self.flatten_nobs_types
          xcurr[:-1,inputnum,self.flatten_input_type_to_range[inputnum,0]:self.flatten_input_type_to_range[inputnum,1]] = movement
      xcurr = np.reshape(xcurr,(xcurr.shape[0]*xcurr.shape[1],xcurr.shape[2]))

    else:
      if self.input_labels:      
        xcurr = np.concatenate((movement,obs[1:,...]),axis=-1)
      else:
        xcurr = obs
    
    
    return xcurr
  
  def get_movement_npad(self):
    npad = compute_npad(self.tspred_global,self.dct_m)
    return npad
  
  def get_predict_mask(self,masksize=None,device=None):
    if masksize is None:
      masksize = self.contextl
      
    if device is None:
      device = self.device
    
    if self.ismasked():
      net_mask = generate_square_full_mask(masksize).to(device)
      is_causal = False
    else:
      net_mask = torch.nn.Transformer.generate_square_subsequent_mask(masksize).to(device)
      is_causal = True
      
    return net_mask,is_causal

  
  def predict_open_loop(self,Xkp,fliespred,scales,burnin,model,maxcontextl=np.inf,debug=False,need_weights=False,nsamples=0):
      """
      Xkp = predict_open_loop(self,Xkp,fliespred,scales,burnin,model,sensory_params,maxcontextl=np.inf,debug=False)

      Args:
        Xkp (ndarray, nkpts x 2 x tpred x nflies): keypoints for all flies for all frames.
        Can be nan for frames/flies to be predicted. Will be overwritten. 
        fliespred (ndarray, nfliespred): indices of flies to predict
        scales (ndarray, nscale x nfliespred): scale parameters for the flies to be predicted
        burnin (int): number of frames to use for initialization
        maxcontextl (int, optional): maximum number of frames to use for context. Default np.inf
        debug (bool, optional): whether to fill in from movement computed from Xkp_all

      Returns:
        Xkp (ndarray, nkpts x 2 x tpred x nflies): keypoints for all flies for all frames,
        with predicted frames/flies filled in. 
        
      Example call:
      res = dataset.predict_open_loop(Xkp,fliespred,scales,burnin,model,maxcontextl=config['contextl'],
                                debug=debug,need_weights=plotattnweights,nsamples=nsamplesfuture)
      """
      model.eval()

      with torch.no_grad():
        w = next(iter(model.parameters()))
        dtype = w.cpu().numpy().dtype
        device = w.device
        
      if nsamples == 0:
        nsamples1 = 1
      else:
        nsamples1 = nsamples
        
      # propagate forward with the 0th sample
      selectsample = 0

      tpred = Xkp.shape[-2]
      nfliespred = len(fliespred)
      # relpose = np.zeros((nsamples1,tpred,nrelative,nfliespred),dtype=dtype)
      # globalpos = np.zeros((nsamples1,tpred,nglobal,nfliespred),dtype=dtype)
      # if self.dct_tau == 0:
      #   ntspred_rel = 1
      # else:
      #   ntspred_rel = self.dct_tau
      # relposefuture = np.zeros((nsamples1,tpred,ntspred_rel,nrelative,nfliespred),dtype=dtype)
      # globalposfuture = np.zeros((nsamples1,tpred,self.ntspred_global,nglobal,nfliespred),dtype=dtype)
      # relposefuture[:] = np.nan
      # globalposfuture[:] = np.nan
      # sensory = None
      # zinputs = None
      if need_weights:
        attn_weights = [None,]*tpred

      if debug:
        labels_true = []
        for i,fly in enumerate(fliespred):
          # compute the pose for pred flies for first burnin frames
          labelscurr = PoseLabels(Xkp=Xkp[...,fly],scale=scales[...,i],
                                  simplify_out=self.simplify_out,
                                  discreteidx=self.discretefeat,
                                  tspred_global=self.tspred_global,
                                  ntspred_relative=self.ntspred_relative,
                                  zscore_params=self.zscore_params,
                                  is_velocity=self.compute_pose_vel,
                                  flatten_labels=self.flatten_labels)
          labels_true.append(labelscurr)
        
      labels = []
      inputs = []
      for i,fly in enumerate(fliespred):
        # compute the pose for pred flies for first burnin frames
        labelscurr = PoseLabels(Xkp=Xkp[...,:burnin,fly],scale=scales[...,i],
                                simplify_out=self.simplify_out,
                                discreteidx=self.discretefeat,
                                tspred_global=self.tspred_global,
                                ntspred_relative=self.ntspred_relative,
                                zscore_params=self.zscore_params,
                                is_velocity=self.compute_pose_vel,
                                flatten_labels=self.flatten_labels)
        labels.append(labelscurr)
        # compute sensory features for first burnin frames
        inputscurr = ObservationInputs(Xkp=Xkp[:,:,1:burnin+1,fly],fly=fly,
                                       zscore_params=self.zscore_params,
                                       simplify_in=self.simplify_in)
        inputs.append(inputscurr)
        
      # movement_true = compute_movement(X=Xkp[...,fliespred],scale=scales,simplify=self.simplify_out,
      #                                  compute_pose_vel=self.compute_pose_vel).transpose((1,0,2)).astype(dtype)
      
      # # outputs -- hide frames past burnin
      # Xkp[:,:,burnin:,fliespred] = np.nan
      
      # # compute the pose for pred flies for first burnin frames
      # relpose0,globalpos0 = compute_pose_features(Xkp[...,:burnin,fliespred],scales)
      # relpose[:,:burnin] = relpose0.transpose((1,0,2))
      # globalpos[:,:burnin] = globalpos0.transpose((1,0,2))
      # # compute one-frame movement for pred flies between first burnin frames 
      # # total length: burnin-1
      # movement0 = compute_movement(relpose=relpose0,
      #                             globalpos=globalpos0,
      #                             simplify=self.simplify_out,
      #                             compute_pose_vel=self.compute_pose_vel)

      # to-do: flattening not yet implemented in PoseLabels
      # movement0 = movement0.transpose((1,0,2))
      # if self.flatten:
      #   zmovement = np.zeros((tpred-1,self.noutput_tokens_per_timepoint,self.flatten_max_doutput,nfliespred),dtype=dtype)
      # else:
      #   zmovement = np.zeros((tpred-1,movement0.shape[1],nfliespred),dtype=dtype)
        
      # for i in range(nfliespred):
      #   zmovementcurr = self.zscore_nextframe_labels(movement0[...,i])
      #   if self.flatten:
      #     if self.discretize:
      #       zmovement_todiscretize = zmovementcurr[...,self.discreteidx]
      #       zmovement_discrete = discretize_labels(zmovement_todiscretize,self.discretize_bin_edges,soften_to_ends=True)
      #       zmovement[:burnin-1,:len(self.discreteidx),:self.discretize_nbins,i] = zmovement_discrete
      #     if self.continuous:
      #       zmovement[:burnin-1,-1,:len(self.continuous_idx),i] = zmovementcurr[...,self.continuous_idx]            
      #   else:
      #     zmovement[:burnin-1,:,i] = zmovementcurr
        
      # # compute sensory features for first burnin frames
      # if self.simplify_in is None:
      #   for i in range(nfliespred):
      #     flynum = fliespred[i]
      #     sensorycurr = compute_sensory_wrapper(Xkp[...,:burnin,:],flynum,
      #                                           theta_main=globalpos[0,:burnin,featthetaglobal,i]) # 0th sample
      #     if sensory is None:
      #       nsensory = sensorycurr.shape[0]
      #       sensory = np.zeros((tpred,nsensory,nfliespred),dtype=dtype)
      #     sensory[:burnin,:,i] = sensorycurr.T
    
      # for i in range(nfliespred):
      #   if self.simplify_in is None:
      #     rawinputscurr = combine_inputs(relpose=relpose[0,:burnin,:,i],
      #                                    sensory=sensory[:burnin,:,i],dim=1)
      #   else:
      #     rawinputscurr = relpose[:burnin,:,i,0]
          
      #   zinputscurr = self.zscore_input(rawinputscurr)
        
        # if self.flatten_obs:
        #   zinputscurr = self.apply_flatten_input(zinputscurr)
        # elif self.flatten:
        #   zinputscurr = zinputscurr[:,None,:]
        # if zinputs is None:
        #   zinputs = np.zeros((tpred,)+zinputscurr.shape[1:]+(nfliespred,),dtype=dtype)
        # zinputs[:burnin,...,i] = zinputscurr
              
      if self.ismasked():
        # to do: figure this out for flattened models
        masktype = 'last'
        dummy = np.zeros((1,self.d_output))
        dummy[:] = np.nan
      else:
        masktype = None
      
      # start predicting motion from frame burnin-1 to burnin = t
      masksizeprev = None
      for t in tqdm.trange(burnin,tpred):
        t0 = int(np.maximum(t-maxcontextl,0))
              
        for i in range(nfliespred):
          flynum = fliespred[i]
          # t should be the last element of inputs
          assert t == inputs[i].ntimesteps
          zinputcurr = inputs[i].get_raw_inputs()
          if self.input_labels:
            assert t == inputs[i].ntimesteps
            zmovementin = labels[i].get_input_labels()
            if self.ismasked():
              # to do: figure this out for flattened model
              zmovementin = np.r_[zmovementin,dummy]
          else:
            zmovementin = None
          # construct_input crops off the first frame of zinputcurr - 
          # zinputcurr should be one frame longer than zmovementin
          xcurr = self.construct_input(zinputcurr,movement=zmovementin)
                    
          # zinputcurr = zinputs[t0:t,...,i] # t-t0 length, last frame corresponds to t-1
          # relposecurr = relpose[0,t-1,:,i] # 0th sample, t-1th frame
          # globalposcurr = globalpos[0,t-1,:,i] # 0th sample, t-1th frame

          
          # if self.input_labels:
          #   zmovementin = zmovement[t0:t-1,...,i] # t-t0-1 length, last frame corresponds to t-2
          #   if self.ismasked():
          #     # to do: figure this out for flattened model
          #     zmovementin = np.r_[zmovementin,dummy]
          # else:
          #   zmovementin = None
          # xcurr = self.construct_input(zinputcurr,movement=zmovementin)
          
          # if self.flatten:
          #     xcurr = np.zeros((zinputcurr.shape[0],self.ntokens_per_timepoint,self.flatten_max_dinput),dtype=dtype)
          #     xcurr[:,:self.flatten_nobs_types,:] = zinputcurr
          #     # movement not set for last time points, will be 0s
          #     xcurr[:-1,self.flatten_nobs_types:,:self.flatten_max_doutput] = zmovementin
          #     xcurr = np.reshape(xcurr,(xcurr.shape[0]*xcurr.shape[1],xcurr.shape[2]))
          #     lastidx = xcurr.shape[0]-self.noutput_tokens_per_timepoint
          #   else:
          #     xcurr = np.concatenate((zmovementin,zinputcurr[1:,...]),axis=-1)
          # else:
          #   xcurr = zinputcurr

          
            
          xcurr = torch.tensor(xcurr)
          xcurr,_,_ = self.mask_input(xcurr,masktype)
          
          if debug:
            zmovementout = np.tile(self.zscore_labels(movement_true[t-1,:,i]).astype(dtype)[None],(nsamples1,1))
          else:
            
            if self.flatten:
              # to do: not sure if multiple samples here works
              
              zmovementout = np.zeros((nsamples1,self.d_output),dtype=dtype)
              zmovementout_flattened = np.zeros((self.noutput_tokens_per_timepoint,self.flatten_max_doutput),dtype=dtype)
              
              for token in range(self.noutput_tokens_per_timepoint):

                lastidx = xcurr.shape[0]-self.noutput_tokens_per_timepoint
                masksize = lastidx+token
                net_mask,is_causal = self.get_predict_mask(masksize=masksize,device=device)

                with torch.no_grad():
                  predtoken = model(xcurr[None,:lastidx+token,...].to(device),mask=net_mask,is_causal=is_causal)
                # to-do: integrate with labels object
                if token < len(self.discreteidx):
                  # sample
                  sampleprob = torch.softmax(predtoken[0,-1,:self.discretize_nbins],dim=-1)
                  binnum = int(weighted_sample(sampleprob,nsamples=nsamples1))
                  
                  # store in input
                  xcurr[lastidx+token,binnum[0]] = 1.
                  zmovementout_flattened[token,binnum[0]] = 1.
                  
                  # convert to continuous
                  nsamples_per_bin = self.discretize_bin_samples.shape[0]
                  sample = int(torch.randint(low=0,high=nsamples_per_bin,size=(nsamples,)))
                  zmovementcurr = self.discretize_bin_samples[sample,token,binnum]
                  
                  # store in output
                  zmovementout[:,self.discreteidx[token]] = zmovementcurr
                else: # else token < len(self.discreteidx)
                  # continuous
                  zmovementout[:,self.continuous_idx] = predtoken[0,-1,:len(self.continuous_idx)].cpu()
                  zmovementout_flattened[token,:len(self.continuous_idx)] = zmovementout[self.continuous_idx,0]
                  
            else: # else flatten

              masksize = t-t0
              if masksize != masksizeprev:
                net_mask,is_causal = self.get_predict_mask(masksize=masksize,device=device)
                masksizeprev = masksize
            
              if need_weights:
                with torch.no_grad():
                  pred,attn_weights_curr = get_output_and_attention_weights(model,xcurr[None,...].to(device),net_mask)
                # dimensions correspond to layer, output frame, input frame
                attn_weights_curr = torch.cat(attn_weights_curr,dim=0).cpu().numpy()
                if i == 0:
                  attn_weights[t] = np.tile(attn_weights_curr[...,None],(1,1,1,nfliespred))
                  attn_weights[t][...,1:] = np.nan
                else:
                  attn_weights[t][...,i] = attn_weights_curr
              else:
                with torch.no_grad():
                  # predict for all frames
                  # masked: movement from 0->1, ..., t->t+1
                  # causal: movement from 1->2, ..., t->t+1
                  # last prediction: t->t+1
                  pred = model.output(xcurr[None,...].to(device),mask=net_mask,is_causal=is_causal)
              # to-do: this is not incorportated into sampling, probably should be
              if model.model_type == 'TransformerBestState' or model.model_type == 'TransformerState':
                pred = model.randpred(pred)
              # z-scored movement from t to t+1
              pred = pred_apply_fun(pred,lambda x: x[0,-1,...].cpu())
              labels[i].append_raw(pred)
              #zmovementout = self.get_full_pred(pred,sample=True,nsamples=nsamples)
              #zmovementout = zmovementout.numpy()
            # end else flatten
          # end else debug

          Xkp_next = labels[i].get_next_keypoints(ts=[t,],nsamples=nsamples)
          # if nsamples == 0:
          #   zmovementout = zmovementout[None,...]
          # # relposenext is nsamples x ntspred_relative x nrelative
          # # globalposnext is nsamples x ntspred_global x nglobal
          # relposenext,globalposnext = self.pred2pose(relposecurr,globalposcurr,zmovementout)
          # relpose[:,t,:,i] = relposenext[:,0] # select next time point, all samples, all features
          # globalpos[:,t,:,i] = globalposnext[:,0]
          # # relposefuture is (nsamples1,tpred,ntspred_rel,nrelative,nfliespred)
          # relposefuture[:,t,:,:,i] = relposenext
          # globalposfuture[:,t,:,:,i] = globalposnext
          # if self.flatten:
          #   zmovement[t-1,:,:,i] = zmovementout_flattened
          # else:
          #   zmovement[t-1,:,i] = zmovementout[selectsample,self.nextframeidx]
          # # next frame
          # featnext = combine_relative_global(relposenext[selectsample,0,:],globalposnext[selectsample,0,:])
          # Xkp_next = mabe.feat2kp(featnext,scales[...,i])
          # Xkp_next = Xkp_next[:,:,0,0]
          Xkp[:,:,t,flynum] = Xkp_next
          
          # we could probably save a little computation by using labels here
          inputs[i].append_keypoints(Xkp[:,:,t,:],fly=flynum,scale=scales[...,i])
          
        # end loop over flies

        # if self.simplify_in is None:
        #   for i in range(nfliespred):
        #     flynum = fliespred[i]
        #     sensorynext = compute_sensory_wrapper(Xkp[...,[t,],:],flynum,
        #                                           theta_main=globalpos[selectsample,[t,],featthetaglobal,i])
        #     sensory[t,:,i] = sensorynext.T
    
        # for i in range(nfliespred):
        #   if self.simplify_in is None:
        #     rawinputsnext = combine_inputs(relpose=relpose[selectsample,[t,],:,i],
        #                                   sensory=sensory[[t,],:,i],dim=1)
        #   else:
        #     rawinputsnext = relpose[selectsample,[t,],:,i]
        #   zinputsnext = self.zscore_input(rawinputsnext)         
        #   zinputs[t,...,i] = zinputsnext
        # end loop over flies
      # end loop over time points

      # if self.flatten:
      #   if self.flatten_obs:
      #     zinputs_unflattened = np.zeros((zinputs.shape[0],self.dfeat,nfliespred))
      #     for i,v in enumerate(self.flatten_obs_idx.values()):
      #       zinputs_unflattened[:,v[0]:v[1],:] = zinputs[:,i,:self.flatten_dinput_pertype[i],:]
      #     zinputs = zinputs_unflattened
      #   else:
      #     zinputs = zinputs[:,0,...]

      if need_weights:
        return Xkp,inputs,labels,attn_weights
      else:
        return Xkp,inputs,labels
  
  def get_movement_names_global(self):
    return self.data[0].labels.get_nextglobal_names()

  def get_movement_names(self):
    return self.data[0].labels.get_nextcossin_names()
  
  def get_outnames(self):
    """
    outnames = self.get_outnames()

    Returns:
        outnames (list of strings): names of each output motion
    """
    return self.data[0].labels.get_multi_names()
    
  # TODO REMOVE THIS
  def parse_label_fields(self,example):
    
    labels_discrete = None
    labels_todiscretize = None
    labels_stacked = None
    
    # get labels_continuous, labels_discrete from example
    if isinstance(example,dict):
      if 'labels' in example:
        labels_continuous = example['labels']
      elif 'continuous' in example:
        labels_continuous = example['continuous'] # prediction
      else:
        raise ValueError('Could not find continuous labels')
      if 'labels_discrete' in example:
        labels_discrete = example['labels_discrete']
      elif 'discrete' in example:
        labels_discrete = example['discrete']
      if 'labels_todiscretize' in example:
        labels_todiscretize = example['labels_todiscretize']
      if 'labels_stacked' in example:
        labels_stacked = example['labels_stacked']      
    else:
      labels_continuous = example
    if self.flatten:
      labels_continuous,labels_discrete = self.unflatten_labels(labels_continuous)
          
    return labels_continuous,labels_discrete,labels_todiscretize,labels_stacked

  def unflatten_labels(self,labels_flattened):
    assert self.flatten_labels
    sz = labels_flattened.shape
    newsz = sz[:-2]+(self.ntimepoints,self.ntokens_per_timepoint,self.flatten_max_doutput)
    if not self.ismasked():
      pad = torch.zeros(sz[:-2]+(1,self.flatten_max_doutput),dtype=labels_flattened.dtype,device=labels_flattened.device)
      labels_flattened = torch.cat((pad,labels_flattened),dim=-2)
    labels_flattened = labels_flattened.reshape(newsz)
    if self.d_output_continuous > 0:
      labels_continuous = labels_flattened[...,-1,:self.d_output_continuous]
    else:
      labels_continuous = None
    if self.discretize:
      labels_discrete = labels_flattened[...,self.flatten_nobs_types:,:self.discretize_nbins]
      if self.continuous:
        labels_discrete = labels_discrete[...,:-1,:]
    else:
      labels_discrete = None
    return labels_continuous,labels_discrete
    
  def apply_flatten_input(self,input):
    
    if type(input) == np.ndarray:
      input = torch.Tensor(input)
    
    if self.flatten_obs == False:
      return input
    
    # input is of size ...,contextl,d_input
    sz = input.shape[:-2]
    contextl = input.shape[-2]
    newinput = torch.zeros(sz+(contextl,self.flatten_nobs_types,self.flatten_max_dinput),dtype=input.dtype)

    for i,v in enumerate(self.flatten_obs_idx.values()):
      newinput[...,i,:self.flatten_dinput_pertype[i]] = input[...,v[0]:v[1]]
    return newinput
    
  def unflatten_input(self,input_flattened):
    assert self.flatten_obs
    sz = input_flattened.shape
    if not self.ismasked():
      pad = torch.zeros(sz[:-2]+(1,self.flatten_dinput),dtype=input_flattened.dtype,device=input_flattened.device)
      input_flattened = torch.cat((input_flattened,pad),dim=-2)      
    resz = sz[:-2]+(self.ntimepoints,self.ntokens_per_timepoint,self.flatten_dinput)
    input_flattened = input_flattened.reshape(resz)
    newsz = sz[:-2]+(self.ntimepoints,self.dfeat)
    newinput = torch.zeros(newsz,dtype=input_flattened.dtype)
    for i,v in enumerate(self.flatten_obs_idx.values()):
      newinput[...,:,v[0]:v[1]] = input_flattened[...,i,self.flatten_input_type_to_range[i,0]:self.flatten_input_type_to_range[i,1]]
    return newinput
  
  def get_full_inputs(self,example=None,idx=None,use_stacked=False):
    if example is None:
      example = self[idx]
    if self.flatten_obs:
      if use_stacked and \
        ('input_stacked' in example and example['input_stacked'] is not None):
        return example['input_stacked']
      else:
        return self.unflatten_input(example['input'])
    else:
      return self.remove_labels_from_input(example['input'])
        
  def get_continuous_discrete_labels(self,example):

    # get labels_continuous, labels_discrete from example
    labels_continuous,labels_discrete,_,_ = self.parse_label_fields(example)      
    return labels_continuous,labels_discrete
        
  def get_continuous_labels(self,example):

    labels_continuous,_ = self.get_continuous_discrete_labels(example)
    return labels_continuous
  
  def get_discrete_labels(self,example):
    _,labels_discrete = self.get_continuous_discrete_labels(example)

    return labels_discrete
  
  def get_full_pred(self,pred,**kwargs):
    return self.get_full_labels(example=pred,ispred=True,**kwargs)
        
  def convert_cos_sin_to_angle(self,movements_in):
    # relpose_cos_sin = WORKING HERE
    if self.compute_pose_vel:
      return movements_in.copy()
    relpose_cos_sin = movements_in[...,-self.nrelrep:]
    relpose_angle = relpose_cos_sin_to_angle(relpose_cos_sin,discreteidx=self.discretefeat)
    return np.concatenate((movements_in[...,:-self.nrelrep],relpose_angle),axis=-1)
        
  def get_full_labels(self,example=None,idx=None,use_todiscretize=False,sample=False,use_stacked=False,ispred=False,nsamples=0):
    
    if self.discretize and sample:
      return self.sample_full_labels(example=example,idx=idx,nsamples=nsamples)
    
    if example is None:
      example = self[idx]

    # get labels_continuous, labels_discrete from example
    labels_continuous,labels_discrete,labels_todiscretize,labels_stacked = \
      self.parse_label_fields(example)
      
    if self.flatten_labels:
      if use_stacked and labels_stacked is not None:
        labels_continuous,labels_discrete = self.unflatten_labels(labels_stacked)
          
    if self.discretize:
      # should be ... x d_output_discrete x discretize_nbins
      sz = labels_discrete.shape
      newsz = sz[:-2]+(self.d_output,)
      labels = torch.zeros(newsz,dtype=labels_discrete.dtype)
      if self.d_output_continuous > 0:
        labels[...,self.continuous_idx] = labels_continuous
      if use_todiscretize and (labels_todiscretize is not None):
        labels[...,self.discreteidx] = labels_todiscretize
      else:
        labels[...,self.discreteidx] = labels_discrete_to_continuous(labels_discrete,
                                                                      torch.tensor(self.discretize_bin_edges))
    else:
      labels = labels_continuous.clone()
        
    return labels
  
  def sample_full_labels(self,example=None,idx=None,nsamples=0):
    if example is None:
      example = self[idx]
      
    nsamples1 = nsamples
    if nsamples1 == 0:
      nsamples1 = 1
      
    # get labels_continuous, labels_discrete from example
    labels_continuous,labels_discrete,_,_ = self.parse_label_fields(example)
      
    if not self.discretize:
      return labels_continuous
    
    # should be ... x d_output_continuous
    sz = labels_discrete.shape[:-2]
    dtype = labels_discrete.dtype
    newsz = (nsamples1,)+sz+(self.d_output,)
    labels = torch.zeros(newsz,dtype=dtype)
    if self.continuous:
      labels[...,self.continuous_idx] = labels_continuous
    
    # labels_discrete is ... x nfeat x nbins
    nfeat = labels_discrete.shape[-2]
    nbins = labels_discrete.shape[-1]
    szrest = labels_discrete.shape[:-2]
    if len(szrest) == 0:
      n = 1
    else:
      n = np.prod(szrest)
    nsamples_per_bin = self.discretize_bin_samples.shape[0]
    for i in range(nfeat):
      binnum = weighted_sample(labels_discrete[...,i,:].reshape((n,nbins)),nsamples=nsamples)
      sample = torch.randint(low=0,high=nsamples_per_bin,size=(nsamples,n))
      labelscurr = torch.Tensor(self.discretize_bin_samples[sample,i,binnum].reshape((nsamples,)+szrest))
      labels[...,self.discreteidx[i]] = labelscurr
      
    if nsamples == 0:
      labels = labels[0]

    return labels
      
def get_batch_idx(example,idx):
  
  if isinstance(example,np.ndarray) or torch.is_tensor(example):
    return example[idx,...]
  
  example1 = {}
  for kw,v in example.items():
    if isinstance(v,np.ndarray) or torch.is_tensor(v):
      example1[kw] = v[idx,...]
    elif isinstance(v,dict):
      example1[kw] = get_batch_idx(v,idx)

  return example1

lossfcn_discrete = torch.nn.CrossEntropyLoss()
lossfcn_continuous = torch.nn.L1Loss()

def causal_criterion(tgt,pred):
  d = tgt.shape[-1]
  err = torch.sum(torch.abs(tgt-pred))/d
  return err

def mixed_causal_criterion(tgt,pred,weight_discrete=.5,extraout=False):
  iscontinuous = tgt['labels'] is not None
  isdiscrete = tgt['labels_discrete'] is not None

  if iscontinuous:
    n = np.prod(tgt['labels'].shape[:-1])
  else:
    n = np.prod(tgt['labels_discrete'].shape[:-2])
  if iscontinuous:
    err_continuous = lossfcn_continuous(pred['continuous'],tgt['labels'].to(device=pred['continuous'].device))*n
  else:
    err_continuous = torch.tensor(0.,dtype=tgt['labels_discrete'].dtype,device=tgt['labels_discrete'].device)
  if isdiscrete:
    pd = pred['discrete']
    newsz = (np.prod(pd.shape[:-1]),pd.shape[-1])
    pd = pd.reshape(newsz)
    td = tgt['labels_discrete'].to(device=pd.device).reshape(newsz)
    err_discrete = lossfcn_discrete(pd,td)*n
  else:
    err_discrete = torch.tensor(0.,dtype=tgt['labels'].dtype,device=tgt['labels'].device)
  err = (1-weight_discrete)*err_continuous + weight_discrete*err_discrete
  if extraout:
    return err,err_discrete,err_continuous
  else:
    return err
  
def dct_consistency(pred):
  return

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

def mixed_masked_criterion(tgt,pred,mask,device,weight_discrete=.5,extraout=False):
  n = torch.count_nonzero(mask)
  err_continuous = lossfcn_continuous(pred['continuous'][mask,:],tgt['labels'].to(device=device)[mask,:])*n
  err_discrete = lossfcn_discrete(pred['discrete'][mask,...],tgt['labels_discrete'].to(device=device)[mask,...])*n
  err = (1-weight_discrete)*err_continuous + weight_discrete*err_discrete
  if extraout:
    return err,err_discrete,err_continuous
  else:
    return err

######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(torch.nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000,
               ntokens_per_timepoint: int = 1):
    super().__init__()

    # during training, randomly zero some of the inputs with probability p=dropout
    self.dropout = torch.nn.Dropout(p=dropout)

    pe = torch.zeros(1,max_len,d_model)
    position = torch.arange(max_len).unsqueeze(1)
    
    # if many tokens per time point, then have a one-hot encoding of token type
    if ntokens_per_timepoint > 1:
      nwave = d_model-ntokens_per_timepoint
      for i in range(ntokens_per_timepoint):
        pe[0,:,nwave+i] = 2*((position[:,0] % ntokens_per_timepoint)==i).to(float)-1
    else:
      nwave = d_model
      
    # compute sine and cosine waves at different frequencies
    # pe[0,:,i] will have a different value for each word (or whatever)
    # will be sines for even i, cosines for odd i,
    # exponentially decreasing frequencies with i
    div_term = torch.exp(torch.arange(0,nwave,2)*(-math.log(10000.0)/nwave))
    nsinwave = int(np.ceil(nwave/2))
    ncoswave = nwave-nsinwave
    pe[0,:,0:nwave:2] = torch.sin(position * div_term[:nsinwave])
    pe[0,:,1:nwave:2] = torch.cos(position * div_term[:ncoswave])

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

  def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, is_causal: bool = False) -> torch.Tensor:
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
    transformer_output = self.transformer_encoder(src,mask=src_mask,is_causal=is_causal)

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

  def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, is_causal: bool = False) -> torch.Tensor:
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
    transformer_output = self.transformer_encoder(src,mask=src_mask,is_causal=is_causal)

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
  
class myTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
  
  def __init__(self,*args,need_weights=False,**kwargs):
    super().__init__(*args,**kwargs)
    self.need_weights = need_weights
  
  def _sa_block(self, x: torch.Tensor,
                attn_mask: typing.Optional[torch.Tensor], 
                key_padding_mask: typing.Optional[torch.Tensor],
                is_causal: bool = False) -> torch.Tensor:
    x = self.self_attn(x, x, x,
                       attn_mask=attn_mask,
                       key_padding_mask=key_padding_mask,
                       need_weights=self.need_weights,
                       is_causal=is_causal)[0]
    return self.dropout1(x)
  def set_need_weights(self,need_weights):
    self.need_weights = need_weights
  
class TransformerModel(torch.nn.Module):

  def __init__(self, d_input: int, d_output: int,
               d_model: int = 2048, nhead: int = 8, d_hid: int = 512,
               nlayers: int = 12, dropout: float = 0.1,
               ntokens_per_timepoint: int = 1,
               input_idx = None, input_szs = None, embedding_types = None, embedding_params = None,
               d_output_discrete = None, nbins = None,
               ):
    super().__init__()
    self.model_type = 'Transformer'

    self.is_mixed = nbins is not None
    if self.is_mixed:
      self.d_output_continuous = d_output
      self.d_output_discrete = d_output_discrete
      self.nbins = nbins
      d_output = self.d_output_continuous + self.d_output_discrete*self.nbins

    # frequency-based representation of word position with dropout
    self.pos_encoder = PositionalEncoding(d_model,dropout,ntokens_per_timepoint=ntokens_per_timepoint)

    # create self-attention + feedforward network module
    # d_model: number of input features
    # nhead: number of heads in the multiheadattention models
    # dhid: dimension of the feedforward network model
    # dropout: dropout value
    encoder_layers = myTransformerEncoderLayer(d_model,nhead,d_hid,dropout,batch_first=True)

    # stack of nlayers self-attention + feedforward layers
    # nlayers: number of sub-encoder layers in the encoder
    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers,nlayers)

    # encoder and decoder are currently not tied together, but maybe they should be? 
    # fully-connected layer from input size to d_model
    
    if input_idx is not None:
      self.encoder = ObsEmbedding(d_model=d_model,input_idx=input_idx,input_szs=input_szs,
                                  embedding_types=embedding_types,embedding_params=embedding_params)
    else:
      self.encoder = torch.nn.Linear(d_input,d_model)

    # fully-connected layer from d_model to input size
    self.decoder = torch.nn.Linear(d_model,d_output)

    # store hyperparameters
    self.d_model = d_model

    self.init_weights()

  def init_weights(self) -> None:
    pass

  def forward(self, src: torch.Tensor, mask: torch.Tensor = None, is_causal: bool = False) -> torch.Tensor:
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
    output = self.transformer_encoder(src,mask=mask,is_causal=is_causal)

    # project back to d_input space
    output = self.decoder(output)
    
    if self.is_mixed:
      output_continuous = output[...,:self.d_output_continuous]
      output_discrete = output[...,self.d_output_continuous:].reshape(output.shape[:-1]+(self.d_output_discrete,self.nbins))
      output = {'continuous': output_continuous, 'discrete': output_discrete}

    return output
  
  def set_need_weights(self,need_weights):
    for layer in self.transformer_encoder.layers:
      layer.set_need_weights(need_weights)

  def output(self,*args,**kwargs):
    
    output = self.forward(*args,**kwargs)
    if self.is_mixed:
      output['discrete'] = torch.softmax(output['discrete'],dim=-1)
    
    return output
  
class ObsEmbedding(torch.nn.Module):
  def __init__(self, d_model: int, input_idx, input_szs, embedding_types, embedding_params):

    super().__init__()

    assert input_idx is not None
    assert input_szs is not None
    assert embedding_types is not None
    assert embedding_params is not None

    self.input_idx = input_idx
    self.input_szs = input_szs

    self.encoder_dict = torch.nn.ModuleDict()
    for k in input_idx.keys():
      emb = embedding_types.get(k,'fc')
      params = embedding_params.get(k,{})
      szcurr = input_szs[k]
      if emb == 'conv1d_feat':
        if len(szcurr) < 2:
          input_channels = 1
        else:
          input_channels = szcurr[1]
        channels = [input_channels,]+params['channels']
        params = {k1:v for k1,v in params.items() if k1 != 'channels'}
        encodercurr = ResNet1d(channels,d_model,d_input=szcurr[0],no_input_channels=True,single_output=True,transpose=False,**params)
      elif emb == 'fc':
        encodercurr = torch.nn.Linear(szcurr[0],d_model)
      elif emb == 'conv1d_time':
        assert(len(szcurr) == 1)
        input_channels = szcurr[0]
        channels = [input_channels,]+params['channels']
        params = {k1:v for k1,v in params.items() if k1 != 'channels'}
        encodercurr = ResNet1d(channels,d_model,no_input_channels=False,single_output=False,transpose=True,**params)
      elif emb == 'conv2d':
        assert(len(szcurr) <= 2)
        if len(szcurr) > 1:
          input_channels = szcurr[1]
          no_input_channels = False
        else:
          input_channels = 1
          no_input_channels = True
        channels = [input_channels,]+params['channels']
        params = {k1:v for k1,v in params.items() if k1 != 'channels'}
        encodercurr = ResNet2d(channels,d_model,no_input_channels=no_input_channels,d_input=szcurr,single_output=True,transpose=True,**params)
      else:
        # consider adding graph networks
        raise ValueError(f'Unknown embedding type {emb}')
      self.encoder_dict[k] = encodercurr
      
  def forward(self,src):
    src = unpack_input(src,self.input_idx,self.input_szs)
    out = 0.
    for k,v in src.items():
      out += self.encoder_dict[k](v)
    return out
  
class Conv1d_asym(torch.nn.Conv1d):
  def __init__(self, *args, padding='same', **kwargs):
    self.padding_off = [0,0]
    padding_sym = padding
    if (type(padding) == tuple) or (type(padding) == list):
      padding_sym = int(np.max(padding))
      for j in range(2):
        self.padding_off[j] = padding_sym - padding[j]
    super().__init__(*args,padding=padding_sym,**kwargs)

  def asymmetric_crop(self,out):
    out = out[...,self.padding_off[0]:out.shape[-1]-self.padding_off[1]]
    return out

  def forward(self,x,*args,**kwargs):
    out = super().forward(x,*args,**kwargs)
    out = self.asymmetric_crop(out)
    return out    
  
class Conv2d_asym(torch.nn.Conv2d):
  def __init__(self, *args, padding='same', **kwargs):
    self.padding_off = [[0,0],[0,0]]
    padding_sym = padding
    if (type(padding) == tuple) or (type(padding) == list):
      padding_sym = list(padding_sym)
      for i in range(2):
        if type(padding[i]) != int:
          padding_sym[i] = int(np.max(padding[i]))
          for j in range(2):
            self.padding_off[i][j] = padding_sym[i] - padding[i][j]
      padding_sym = tuple(padding_sym)
    super().__init__(*args,padding=padding_sym,**kwargs)

  def asymmetric_crop(self,out):
    out = out[...,self.padding_off[0][0]:out.shape[-2]-self.padding_off[0][1],self.padding_off[1][0]:out.shape[-1]-self.padding_off[1][1]]
    return out

  def forward(self,x,*args,**kwargs):
    out = super().forward(x,*args,**kwargs)
    out = self.asymmetric_crop(out)
    return out
  
class ResidualBlock1d(torch.nn.Module):
  
  def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, dilation = 1, padding = 'same', padding_mode='zeros'):
    super().__init__()
    
    self.padding = padding
    self.kernel_size = kernel_size
    self.stride = stride
    self.dilation = dilation
    self.conv1 = torch.nn.Sequential(
      Conv1d_asym(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, padding_mode=padding_mode, dilation=self.dilation),
      torch.nn.BatchNorm1d(out_channels),
      torch.nn.ReLU()
    )
    self.conv2 = torch.nn.Sequential(
      Conv1d_asym(out_channels, out_channels, kernel_size=self.kernel_size, stride=1, padding=self.padding, padding_mode=padding_mode, dilation=self.dilation),
      torch.nn.BatchNorm1d(out_channels)
    )
    if (in_channels != out_channels) or (self.stride > 1):
      self.downsample = torch.nn.Sequential(
        torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
        torch.nn.BatchNorm1d(out_channels)
      )
    else:
      self.downsample = None
    self.relu = torch.nn.ReLU()
    self.out_channels = out_channels
      
  def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.conv2(out)
    if self.downsample:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)
    return out
  
  def compute_output_shape(self,din):
    if type(self.padding) == str:
      if self.padding == 'same':
        return (self.out_channels,din)
      elif self.padding == 'valid':
        padding = 0
      else:
        raise ValueError(f'Unknown padding type {self.padding}')
    if len(self.padding) == 1:
      padding = 2*self.padding
    else:
      padding = np.sum(self.padding)
    dout1 = np.floor((din + padding - self.dilation*(self.kernel_size-1)-1)/self.stride+1)
    dout = (dout1 + padding - self.dilation*(self.kernel_size-1)-1)+1
    sz = (self.out_channels,int(dout))
    return sz
  
class ResidualBlock2d(torch.nn.Module):
  
  def __init__(self, in_channels, out_channels, kernel_size = (3,3), stride = (1,1), dilation = (1,1), padding = 'same', padding_mode = 'zeros'):
    super().__init__()

    self.padding = padding          
    self.kernel_size = kernel_size
    self.stride = stride
    self.dilation = dilation
    self.conv1 = torch.nn.Sequential(
      Conv2d_asym(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, padding_mode=padding_mode, dilation=self.dilation),
      torch.nn.BatchNorm2d(out_channels),
      torch.nn.ReLU()
    )
    self.conv2 = torch.nn.Sequential(
      Conv2d_asym(out_channels, out_channels, kernel_size=self.kernel_size, stride=1, padding=self.padding, padding_mode=padding_mode, dilation=self.dilation),
      torch.nn.BatchNorm2d(out_channels)
    )
    if (in_channels != out_channels) or (np.any(np.array(self.stride) > 1)):
      self.downsample = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
        torch.nn.BatchNorm2d(out_channels)
      )
    else:
      self.downsample = None
    self.relu = torch.nn.ReLU()
    self.out_channels = out_channels
    
  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.conv2(out)
    if self.downsample:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)
    return out
  
  def compute_output_shape(self,din):
    if type(self.padding) == str:
      if self.padding == 'same':
        return (self.out_channels,)+din
      elif self.padding == 'valid':
        padding = (0,0)
      else:
        raise ValueError(f'Unknown padding type {self.padding}')

    if len(self.padding) == 1:
      padding = [self.padding,self.padding]
    else:
      padding = self.padding
    padding = np.array(padding)
    paddingsum = np.zeros(2,dtype=int)
    for i in range(2):
      if len(padding[i]) == 1:
        paddingsum[i] = 2*padding[i]
      else:
        paddingsum[i] = int(np.sum(padding[i]))
    dout1 = np.floor((np.array(din) + paddingsum - np.array(self.dilation)*(np.array(self.kernel_size)-1)-1)/np.array(self.stride)+1).astype(int)
    dout = ((dout1 + paddingsum - np.array(self.dilation)*(np.array(self.kernel_size)-1)-1)+1).astype(int)
    sz = (self.out_channels,) + tuple(dout)
    return sz
  
class ResNet1d(torch.nn.Module):
  def __init__(self,channels,d_output,d_input=None,no_input_channels=False,single_output=False,transpose=False,**kwargs):
    super().__init__()
    self.channels = channels
    self.d_output = d_output
    self.d_input = d_input
    self.no_input_channels = no_input_channels
    self.transpose = transpose
    self.single_output = single_output
    
    if no_input_channels:
      assert channels[0] == 1
    
    nblocks = len(channels)-1
    self.layers = torch.nn.ModuleList()
    sz = (channels[0],d_input)
    for i in range(nblocks):
      self.layers.append(ResidualBlock1d(channels[i],channels[i+1],**kwargs))
      if d_input is not None:
        sz = self.layers[-1].compute_output_shape(sz[-1])
    if single_output:
      if d_input is None:
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(channels[-1],d_output)
      else:
        self.avg_pool = None
        self.fc = torch.nn.Linear(int(np.prod(sz)),d_output)
    else:
      self.avg_pool = None
      self.fc = torch.nn.Conv1d(channels[-1],d_output,1)
      
  def forward(self,x):

    if self.transpose and not self.no_input_channels:
      x = x.transpose(-1,-2)

    if self.no_input_channels:
      dim = -1
    else:
      dim = -2
    
    sz0 = x.shape
    d_input = sz0[-1]

    if self.single_output and (self.d_input is not None):
      assert d_input == self.d_input
    
    sz = (int(np.prod(sz0[:dim])),self.channels[0],d_input)
    x = x.reshape(sz)
    
    for layer in self.layers:
      x = layer(x)
    if self.avg_pool is not None:
      x = self.avg_pool(x)
    if self.single_output:
      x = torch.flatten(x,1)
    x = self.fc(x)
    
    if self.single_output:
      dimout = -1
    else:
      dimout = -2
    x = x.reshape(sz0[:dim]+x.shape[dimout:])
    
    if self.transpose and not self.single_output:
      x = x.transpose(-1,-2)

    return x
  
class ResNet2d(torch.nn.Module):
  def __init__(self,channels,d_output,d_input=None,no_input_channels=False,single_output=False,transpose=False,**kwargs):
    super().__init__()
    self.channels = channels
    self.d_output = d_output
    self.d_input = d_input
    self.no_input_channels = no_input_channels
    self.transpose = transpose
    
    if no_input_channels:
      assert channels[0] == 1
    
    nblocks = len(channels)-1
    self.layers = torch.nn.ModuleList()
    is_d_input = [False,False]
    if d_input is not None:
      if type(d_input) == int:
        d_input (0,d_input)
      elif len(d_input) < 2:
        d_input = (0,)*(2-len(d_input))+d_input
      is_d_input = [d != 0 for d in d_input]
      sz = (channels[0],) + d_input
    for i in range(nblocks):
      self.layers.append(ResidualBlock2d(channels[i],channels[i+1],**kwargs))
      if d_input is not None:
        sz = self.layers[-1].compute_output_shape(sz[1:])
    self.collapse_dim = []
    if single_output:
      if d_input is None:
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(channels[-1],d_output)
        self.collapse_dim = [-2,-1]
      else:
        self.avg_pool = None
        k = [1,1]
        for i in range(2):
          if is_d_input[i]:
            k[i] = sz[i+1]
            self.collapse_dim.append(-2+i)
        self.fc = torch.nn.Conv2d(channels[-1],d_output,k,padding='valid')
    else:
      self.avg_pool = None
      self.fc = torch.nn.Conv2d(channels[-1],d_output,1)
      
  def forward(self,x):

    if self.transpose and not self.no_input_channels:
      x = torch.movedim(x,-1,-3)

    if self.no_input_channels:
      dim = -2
    else:
      dim = -3
    
    sz0 = x.shape
    d_input = sz0[-2:]
    
    sz = (int(np.prod(sz0[:dim])),self.channels[0])+d_input
    x = x.reshape(sz)
    
    for layer in self.layers:
      x = layer(x)
    if self.avg_pool is not None:
      x = self.avg_pool(x)
      x = torch.flatten(x,1)
      dimout = -1
    else:
      dimout = -3
    x = self.fc(x)
    x = x.reshape(sz0[:dim]+x.shape[dimout:])
    dim_channel = len(sz0[:dim])
    x = torch.squeeze(x,self.collapse_dim)
    
    if self.transpose:
      x = torch.movedim(x,dim_channel,-1)

    return x
    
class DictSum(torch.nn.Module):
  def __init__(self,moduledict):
    super().__init__()
    self.moduledict = moduledict
  def forward(self,x):
    out = 0.
    for k,v in x.items():
      out += self.moduledict[k](v)
    return out

# deprecated, here for backward compatibility
class TransformerMixedModel(TransformerModel):
  
  def __init__(self, d_input: int, d_output_continuous: int = 0,
               d_output_discrete: int = 0, nbins: int = 0,
               **kwargs):
    self.d_output_continuous = d_output_continuous
    self.d_output_discrete = d_output_discrete
    self.nbins = nbins
    d_output = d_output_continuous + d_output_discrete*nbins
    assert d_output > 0
    super().__init__(d_input,d_output,**kwargs)
    
  def forward(self, src: torch.Tensor, mask: torch.Tensor = None, is_causal: bool = False) -> dict:
    output_all = super().forward(src,mask=mask,is_causal=is_causal)
    output_continuous = output_all[...,:self.d_output_continuous]
    output_discrete = output_all[...,self.d_output_continuous:].reshape(output_all.shape[:-1]+(self.d_output_discrete,self.nbins))
    return {'continuous': output_continuous, 'discrete': output_discrete}
  
  def output(self,*args,**kwargs):
    output = self.forward(*args,**kwargs)
    output['discrete'] = torch.softmax(output['discrete'],dim=-1)
    return output
  
def generate_square_full_mask(sz: int) -> torch.Tensor:
  """
  Generates an zero matrix. All words allowed.
  """
  return torch.zeros(sz,sz)

def get_output_and_attention_weights(model, inputs, mask=None, is_causal=False):  

  # set need_weights to True for this function call
  model.set_need_weights(True)

  # where attention weights will be stored, one list element per layer
  activation = [None,]*model.transformer_encoder.num_layers
  def get_activation(layer_num):
    # the hook signature
    def hook(model, inputs, output):
      # attention weights are the second output
      activation[layer_num] = output[1]
    return hook

  # register the hooks
  hooks = [None,]*model.transformer_encoder.num_layers
  for i,layer in enumerate(model.transformer_encoder.layers):
    hooks[i] = layer.self_attn.register_forward_hook(get_activation(i))

  # call the model
  with torch.no_grad():
    output = model.output(inputs, mask=mask, is_causal=is_causal)
  
  # remove the hooks    
  for hook in hooks:
    hook.remove()

  # return need_weights to False
  model.set_need_weights(False)    

  return output,activation
  
def compute_attention_weight_rollout(w0):
  # w0 is nlayers x T x T x ...
  w = np.zeros(w0.shape,dtype=w0.dtype)
  wcurr = np.ones(list(w0.shape)[1:],dtype=w0.dtype)
  # I = np.eye(w0.shape[1],dtype=w0.dtype)
  # sz = np.array(w0.shape[1:])
  # sz[2:] = 1
  # I = I.reshape(sz)

  for i in range(w0.shape[0]):
    wcurr = wcurr * (w0[i,...])
    z = np.maximum(np.sum(wcurr,axis=0,keepdims=True),np.finfo(w0.dtype).eps)
    wcurr = wcurr / z
    w[i,...]  = wcurr
  return w

def save_model(savefile,model,lr_optimizer=None,scheduler=None,loss=None,config=None):
  tosave = {'model':model.state_dict()}
  if lr_optimizer is not None:
    tosave['lr_optimizer'] = lr_optimizer.state_dict()
  if scheduler is not None:
    tosave['scheduler'] = scheduler.state_dict()
  if loss is not None:
    tosave['loss'] = loss
  if config is not None:
    tosave['config'] = config
  tosave['SENSORY_PARAMS'] = SENSORY_PARAMS
  torch.save(tosave,savefile)
  return

def load_model(loadfile,model,device,lr_optimizer=None,scheduler=None,config=None):
  print(f'Loading model from file {loadfile}...')
  state = torch.load(loadfile, map_location=device)
  if model is not None:
    model.load_state_dict(state['model'])
  if lr_optimizer is not None and ('lr_optimizer' in state):
    lr_optimizer.load_state_dict(state['lr_optimizer'])
  if scheduler is not None and ('scheduler' in state):
    scheduler.load_state_dict(state['scheduler'])
  if config is not None:
    load_config_from_model_file(config=config,state=state)
      
  loss = {}
  if 'loss' in state:
    if isinstance(loss,dict):
      loss = state['loss']
    else:
      # backwards compatible
      loss['train'] = loss
      if 'val_loss' in state:
        loss['val'] = state['val_loss']
  return loss

def load_config_from_model_file(loadmodelfile=None,config=None,state=None,no_overwrite=[]):
  if state is None:
    assert loadmodelfile is not None
    print(f'Loading config from file {loadmodelfile}...')
    state = torch.load(loadmodelfile)
  if config is not None and 'config' in state:
    overwrite_config(config,state['config'],no_overwrite=no_overwrite)
  else:
    print(f'config not stored in model file {loadmodelfile}')
  if 'SENSORY_PARAMS' in state:
    for k,v in state['SENSORY_PARAMS'].items():
      SENSORY_PARAMS[k] = v
  else:
    print(f'SENSORY_PARAMS not stored in model file {loadmodelfile}')
  return

def update_loss_nepochs(loss_epoch,nepochs):
  for k,v in loss_epoch.items():
    if v.numel() < nepochs:
      n = torch.zeros(nepochs-v.numel(),dtype=v.dtype,device=v.device)+torch.nan
      loss_epoch[k] = torch.cat((v,n))      
  return


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


def select_featidx_plot(train_dataset,ntspred_plot,ntsplot_global=None,ntsplot_relative=None):

  if ntsplot_global is None: 
    ntsplot_global = np.minimum(train_dataset.ntspred_global,ntspred_plot)
  if ntsplot_relative is None:
    ntsplot_relative = np.minimum(train_dataset.ntspred_relative,ntspred_plot)
  
  if ntsplot_global == 0:
    tidxplot_global = None
  elif ntsplot_global == 1:
    tidxplot_global = np.zeros((nglobal,1),dtype=int)
  elif ntsplot_global == train_dataset.ntspred_global:
    tidxplot_global = np.tile(np.arange(ntsplot_global,dtype=int)[None,:],(nglobal,1))
  else:
    # choose 0 + a variety of different timepoints for each global feature so that a variety of timepoints are selected
    tidxplot_global = np.concatenate((np.zeros((nglobal,1),dtype=int),
                                      np.round(np.linspace(1,train_dataset.ntspred_global-1,(ntsplot_global-1)*nglobal)).astype(int).reshape(-1,nglobal).T),axis=-1)
  if ntsplot_relative == 0:
    tsplot_relative = None
  elif ntsplot_relative == 1:
    tsplot_relative = np.ones((train_dataset.nrelrep,1),dtype=int)
  elif ntsplot_relative == train_dataset.ntspred_relative:
    tsplot_relative = np.tile(np.arange(ntsplot_relative,dtype=int)[None,:]+1,(train_dataset.nrelrep,1))
  else:
    # choose 0 + a variety of different timepoints for each feature so that a variety of timepoints are selected
    tsplot_relative = np.concatenate((np.zeros((train_dataset.nrelrep,1),dtype=int),
                                      np.round(np.linspace(1,train_dataset.ntspred_relative-1,(ntsplot_relative-1)*nrelative)).astype(int).reshape(-1,train_dataset.nrelrep).T),axis=-1)
  ftidx = []
  for fi,f in enumerate(featglobal):
    for ti in range(ntsplot_global):
      ftidx.append((f,train_dataset.tspred_global[tidxplot_global[fi,ti]]))
  for fi,f in enumerate(np.nonzero(train_dataset.featrelative)[0]):
    for ti in range(ntsplot_relative):
      ftidx.append((f,tsplot_relative[fi,ti]))
  featidxplot = train_dataset.ravel_label_index(ftidx)
  return featidxplot

def debug_plot_batch_traj(example_in,train_dataset,criterion=None,config=None,
                          pred=None,data=None,nsamplesplot=3,
                          h=None,ax=None,fig=None,label_true='True',label_pred='Pred',
                          ntsplot=3,ntsplot_global=None,ntsplot_relative=None):
  
  example,samplesplot = subsample_batch(example_in,nsamples=nsamplesplot,
                                        dataset=train_dataset)
  nsamplesplot = len(example)
    
  true_color = [0,0,0]
  pred_cmap = lambda x: plt.get_cmap("tab10")(x%10)
  
  if train_dataset.ismasked():
    mask = example_in['mask']
  else:
    mask = None
  
  if ax is None:
    fig,ax = plt.subplots(1,nsamplesplot,squeeze=False)
    ax = ax[0,:]

  featidxplot,ftplot = example[0].labels.select_featidx_plot(ntsplot=ntsplot,
                                                             ntsplot_global=ntsplot_global,
                                                             ntsplot_relative=ntsplot_relative)
  for i,iplot in enumerate(samplesplot):
    examplecurr = example[i]
    rawlabelstrue = examplecurr.labels.get_train_labels()
    zmovement_continuous_true = rawlabelstrue['continuous']
    zmovement_discrete_true = rawlabelstrue['discrete']
    
    err_total = None
    maskcurr = examplecurr.labels.get_mask()
    if maskcurr is None:
      maskidx = np.nonzero(maskcurr)[0]
    zmovement_continuous_pred = None
    zmovement_discrete_pred = None
    if pred is not None:
      rawpred = get_batch_idx(pred,iplot)
      if 'continuous' in rawpred:
        zmovement_continuous_pred = rawpred['continuous']
      if 'discrete' in rawpred:
        zmovement_discrete_pred = rawpred['discrete']
        zmovement_discrete_pred = torch.softmax(zmovement_discrete_pred,dim=-1)
      if criterion is not None:
        err_total,err_discrete,err_continuous = criterion_wrapper(rawlabelstrue,rawpred,criterion,train_dataset,config)
      # err_movement = torch.abs(zmovement_true[maskidx,:]-zmovement_pred[maskidx,:])/nmask
      # err_total = torch.sum(err_movement).item()/d

    elif data is not None:
      # for i in range(nsamplesplot):
      #   metadata = example[i].get_train_metadata()
      #   t0 = metadata['t0']
      #   flynum = metadata[flynum]
      #   datakp,id = data['X'][:,:,t0:t0+ntimepoints+1,flynum].transpose(2,0,1)

      
      #t0 = example['metadata']['t0'][iplot].item()
      #flynum = example['metadata']['flynum'][iplot].item()
      pass
    
    mult = 6.
    d = len(featidxplot)
    outnames = examplecurr.labels.get_multi_names()
    contextl = examplecurr.labels.ntimepoints_train

    ax[i].cla()
    
    idx_multi_to_multidiscrete = examplecurr.labels.idx_multi_to_multidiscrete
    idx_multi_to_multicontinuous = examplecurr.labels.idx_multi_to_multicontinuous
    for featii,feati in enumerate(featidxplot):
      featidx = idx_multi_to_multidiscrete[feati]
      if featidx < 0:
        continue
      im = np.ones((train_dataset.discretize_nbins,contextl,3))
      ztrue = zmovement_discrete_true[:,featidx,:].cpu().T
      ztrue = ztrue - torch.min(ztrue)
      ztrue = ztrue / torch.max(ztrue)
      im[:,:,0] = 1.-ztrue
      if pred is not None:
        zpred = zmovement_discrete_pred[:,featidx,:].detach().cpu().T
        zpred = zpred - torch.min(zpred)
        zpred = zpred / torch.max(zpred)
        im[:,:,1] = 1.-zpred
      ax[i].imshow(im,extent=(0,contextl,(featii-.5)*mult,(featii+.5)*mult),origin='lower',aspect='auto')
      
    for featii,feati in enumerate(featidxplot):
      featidx = idx_multi_to_multicontinuous[feati]
      if featidx < 0:
        continue
      ax[i].plot([0,contextl],[mult*featii,]*2,':',color=[.5,.5,.5])
      ax[i].plot(mult*featii + zmovement_continuous_true[:,featidx],'-',color=true_color,label=f'{outnames[feati]}, true')
      if mask is not None:
        ax[i].plot(maskidx,mult*featii + zmovement_continuous_true[maskcurr[:-1],featidx],'o',color=true_color,label=f'{outnames[feati]}, true')

      labelcurr = outnames[feati]
      if pred is not None:
        h = ax[i].plot(mult*featii + zmovement_continuous_pred[:,featidx],'--',label=f'{outnames[feati]}, pred',color=pred_cmap(featii))
        if mask is not None:
          ax[i].plot(maskidx,mult*featii + zmovement_continuous_pred[maskcurr[:-1],featidx],'o',color=pred_cmap(featii),label=f'{outnames[feati]}, pred')
        
    for featii,feati in enumerate(featidxplot):
      labelcurr = outnames[feati]
      ax[i].text(0,mult*(featii+.5),labelcurr,horizontalalignment='left',verticalalignment='top')

    if (err_total is not None):
      if train_dataset.discretize:
        ax[i].set_title(f'Err: {err_total.item(): .2f}, disc: {err_discrete.item(): .2f}, cont: {err_continuous.item(): .2f}')
      else:
        ax[i].set_title(f'Err: {err_total.item(): .2f}')
    ax[i].set_xlabel('Frame')
    ax[i].set_ylabel('Movement')
    ax[i].set_ylim([-mult,mult*d])

  fig.tight_layout()

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

def subsample_batch(example,nsamples=1,samples=None,dataset=None):
  israw = type(example) is dict
  islist = type(example) is list
  if samples is not None:
    nsamples = len(samples)

  if israw:
    batch_size = example['input'].shape[0]
  elif islist:
    batch_size = len(example)
  elif type(example) is FlyExample:
    batch_size = int(np.prod(example.pre_sz))
    if batch_size == 1:
      return [example,],np.arange(1)
  else:
    raise ValueError(f'Unknown type {type(example)}')

  if samples is None:
    nsamples = np.minimum(nsamples,batch_size)
    samples = np.round(np.linspace(0,batch_size-1,nsamples)).astype(int)
  else:
    assert np.max(samples) < batch_size 

  if islist:
    return [example[i] for i in samples],samples

  if israw:
    rawbatch = example
    examplelist = []
    for samplei in samples:
      examplecurr = get_batch_idx(rawbatch,samplei)
      assert dataset is not None
      flyexample = FlyExample(example_in=examplecurr,dataset=dataset)
      examplelist.append(flyexample)
  else:
    examplelist = []
    for samplei in samples:
      examplecurr = example.copy_subindex(idx_pre=samplei)
      examplelist.append(examplecurr)
      
  return examplelist,samples  

def debug_plot_pose(example,train_dataset=None,pred=None,data=None,
                    true_discrete_mode='to_discretize',
                    pred_discrete_mode='sample',
                    ntsplot=5,nsamplesplot=3,h=None,ax=None,fig=None,
                    tsplot=None):
 
  example,samplesplot = subsample_batch(example,nsamples=nsamplesplot,
                                        dataset=train_dataset)


  israwpred = type(pred) is dict
  if israwpred:
    batchpred = pred
    pred = []
    for i,samplei in enumerate(samplesplot):
      predcurr = get_batch_idx(batchpred,samplei)
      flyexample = example[i].copy()
      # just to make sure no data sticks around
      flyexample.labels.erase_labels()
      flyexample.labels.set_prediction(predcurr)
      pred.append(flyexample)
  elif type(pred) is FlyExample:
    pred,_ = subsample_batch(pred,samples=samplesplot)

  nsamplesplot = len(example)
  if pred is not None:
    assert(len(pred) == nsamplesplot)

  contextl = example[0].ntimepoints
  if tsplot is None:
    tsplot = np.round(np.linspace(0,contextl-1,ntsplot)).astype(int)
  else:
    ntsplot = len(tsplot)
      
  if tsplot is None:
    tsplot = np.round(np.linspace(0,contextl-1,ntsplot)).astype(int)
  else:
    ntsplot = len(tsplot)

  if ax is None:
    fig,ax = plt.subplots(nsamplesplot,ntsplot,squeeze=False)

  if h is None:
    h = {'kpt0': [], 'kpt1': [], 'edge0': [], 'edge1': []}
  
  if true_discrete_mode == 'to_discretize':
    true_args = {'use_todiscretize': True}
  elif true_discrete_mode == 'sample':
    true_args = {'sample': True}
  else:
    true_args = {}
    
  if pred_discrete_mode == 'sample':
    pred_args = {'nsamples': 1, 'collapse_samples': True}
  else:
    pred_args = {}
    
  for i in range(nsamplesplot):
    iplot = samplesplot[i]
    examplecurr = example[i]
    Xkp_true = examplecurr.labels.get_next_keypoints(**true_args)
    nametrue = 'Labels'
    #['input'][iplot,0,...].numpy(),
    #                            example['init'][iplot,...].numpy(),
    #                            example['labels'][iplot,...].numpy(),
    #                            example['scale'][iplot,...].numpy())
    #Xkp_true = Xkp_true[...,0]
    t0 = examplecurr.metadata['t0']
    flynum = examplecurr.metadata['flynum']
    if pred is not None:
      predcurr = pred[i]
      Xkp_pred = predcurr.labels.get_next_keypoints(**pred_args)
      namepred = 'Pred'
    elif data is not None:
      Xkp_pred = data['X'][:,:,t0:t0+contextl,flynum].transpose(2,0,1)
      namepred = 'Raw data'
    else:
      Xkp_pred = None
    for key in h.keys():
      if len(h[key]) <= i:
        h[key].append([None,]*ntsplot)
        
    minxy = np.nanmin(np.nanmin(Xkp_true[tsplot,:,:],axis=1),axis=0)
    maxxy = np.nanmax(np.nanmax(Xkp_true[tsplot,:,:],axis=1),axis=0)
    if Xkp_pred is not None:
      minxy_pred = np.nanmin(np.nanmin(Xkp_pred[tsplot,:,:],axis=1),axis=0)
      maxxy_pred = np.nanmax(np.nanmax(Xkp_pred[tsplot,:,:],axis=1),axis=0)
      minxy = np.minimum(minxy,minxy_pred)
      maxxy = np.maximum(maxxy,maxxy_pred)    
    for j in range(ntsplot):
      tplot = tsplot[j]
      if j == 0:
        ax[i,j].set_title(f'fly: {flynum} t0: {t0}')
      else:
        ax[i,j].set_title(f't = {tplot}')
        
      h['kpt0'][i][j],h['edge0'][i][j],_,_,_ = mabe.plot_fly(Xkp_true[tplot,:,:],
                                                             skel_lw=2,color=[0,0,0],
                                                             ax=ax[i,j],hkpt=h['kpt0'][i][j],hedge=h['edge0'][i][j])
      if Xkp_pred is not None:
        h['kpt1'][i][j],h['edge1'][i][j],_,_,_ = mabe.plot_fly(Xkp_pred[tplot,:,:],
                                                              skel_lw=1,color=[0,1,1],
                                                              ax=ax[i,j],hkpt=h['kpt1'][i][j],hedge=h['edge1'][i][j])
        if i == 0 and j == 0:
          ax[i,j].legend([h['edge0'][i][j],h['edge1'][i][j]],[nametrue,namepred])
      ax[i,j].set_aspect('equal')
      # minxy = np.nanmin(Xkp_true[:,:,tplot],axis=0)
      # maxxy = np.nanmax(Xkp_true[:,:,tplot],axis=0)
      # if Xkp_pred is not None:
      #   minxy_pred = np.nanmin(Xkp_pred[:,:,tplot],axis=0)
      #   maxxy_pred = np.nanmax(Xkp_pred[:,:,tplot],axis=0)
      #   minxy = np.minimum(minxy,minxy_pred)
      #   maxxy = np.maximum(maxxy,maxxy_pred)
      ax[i,j].set_xlim([minxy[0],maxxy[0]])
      ax[i,j].set_ylim([minxy[1],maxxy[1]])

  return h,ax,fig

def debug_plot_sample(example_in,dataset=None,nplot=3):

  example,samplesplot = subsample_batch(example_in,nsamples=nplot,dataset=dataset)
  nplot = len(example)

  fig,ax = plt.subplots(nplot,2,squeeze=False)
  
  idx = example[0].inputs.get_sensory_feature_idx()
  inputidxstart = [x[0] - .5 for x in idx.values()]
  inputidxtype = list(idx.keys())
  T = example[0].ntimepoints
    
  for iplot,samplei in enumerate(samplesplot):
    ax[iplot,0].cla()
    ax[iplot,1].cla()
    ax[iplot,0].imshow(example[iplot].inputs.get_raw_inputs(),
                       vmin=-3,vmax=3,cmap='coolwarm',aspect='auto')
    ax[iplot,0].set_title(f'Input {samplei}')
    #ax[iplot,0].set_xticks(inputidxstart)
    for j in range(len(inputidxtype)):
      ax[iplot,0].plot([inputidxstart[j],]*2,[-.5,T-.5],'k-')
      ax[iplot,0].text(inputidxstart[j],T-1,inputidxtype[j],horizontalalignment='left')
    lastidx = list(idx.values())[-1][1]
    ax[iplot,0].plot([lastidx-.5,]*2,[-.5,T-.5],'k-')

    #ax[iplot,0].set_xticklabels(inputidxtype)
    ax[iplot,1].imshow(example[iplot].labels.get_multi(zscored=True),
                       vmin=-3,vmax=3,cmap='coolwarm',aspect='auto')
    ax[iplot,1].set_title(f'Labels {samplei}')
  return fig,ax
  
def stack_batch_list(allx,n=None):
  if len(allx) == 0:
    return []
  xv = torch.cat(allx[:n],dim=0)
  nan = torch.zeros((xv.shape[0],1)+xv.shape[2:],dtype=xv.dtype)
  nan[:] = torch.nan
  xv = torch.cat((xv,nan),dim=1)
  xv = xv.flatten(0,1)
  return xv
  
def len_wrapper(x,defaultlen=None):
  if x is None:
    return defaultlen
  if hasattr(x,'__len__'):
    return len(x)
  return 1
  
def stackhelper(all_pred,all_labels,all_mask,all_pred_discrete,all_labels_discrete,nplot):

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
  predv = predv.flatten(0,2)
  labelsv = torch.cat((labelsv,nan),dim=2)
  labelsv = labelsv.flatten(0,2)
  if maskv is not None:
    maskv = torch.cat((maskv,torch.zeros(s[:-1],dtype=bool)),dim=2)
    maskv = maskv.flatten()
  if len(all_pred_discrete) > 0:
    pred_discretev = torch.stack(all_pred_discrete[:nplot],dim=0)
    s = list(pred_discretev.shape)
    s[2] = 1
    nan = torch.zeros(s,dtype=pred_discretev.dtype)
    nan[:] = torch.nan
    pred_discretev = torch.cat((pred_discretev,nan),dim=2)
    pred_discretev = pred_discretev.flatten(0,2)
  else:
    pred_discretev = None
  if len(all_labels_discrete) > 0:
    pred_discretev = torch.stack(all_labels_discrete[:nplot],dim=0)
    s = list(pred_discretev.shape)
    s[2] = 1
    nan = torch.zeros(s,dtype=pred_discretev.dtype)
    nan[:] = torch.nan
    pred_discretev = torch.cat((pred_discretev,nan),dim=2)
    pred_discretev = pred_discretev.flatten(0,2)
  else:
    pred_discretev = None
  
  return predv,labelsv,maskv,pred_discretev
  
def debug_plot_predictions_vs_labels(all_pred,all_labels,ax=None,
                                     prctile_lim=.1,naxc=1,featidxplot=None,
                                     gaplen=2):
  
  d_output = all_pred[0].d_multi
  predv_cont = np.stack([pred.get_multi() for pred in all_pred],axis=0)
  labelsv_cont = np.stack([label.get_multi() for label in all_labels],axis=0)
  nans = np.zeros((len(all_pred),gaplen,d_output),dtype=all_labels[0].dtype) + np.nan
  predv_cont = np.reshape(np.concatenate((predv_cont,nans),axis=1),(-1,d_output))
  labelsv_cont = np.reshape(np.concatenate((labelsv_cont,nans),axis=1),(-1,d_output))
  if all_labels[0].is_discretized():
    predv_discrete = np.stack([pred.get_multi_discrete() for pred in all_pred],axis=0)
    labelsv_discrete = np.stack([labels.get_multi_discrete() for labels in all_labels],axis=0)
    nans = np.zeros((len(all_pred),gaplen)+predv_discrete.shape[-2:],dtype=all_labels[0].dtype) + np.nan
    predv_discrete = np.reshape(np.concatenate((predv_discrete,nans),axis=1),(-1,)+predv_discrete.shape[-2:])
    labelsv_discrete = np.reshape(np.concatenate((labelsv_discrete,nans),axis=1),(-1,)+labelsv_discrete.shape[-2:])
  
  if featidxplot is None:
    featidxplot = np.arange(d_output)
  nfeat = len(featidxplot)
  
  ismasked = all_labels[0].is_masked()
  if ismasked:
    maskv = np.stack([label.get_mask() for label in all_labels],axis=0)
    nans = np.zeros((len(all_pred),gaplen),dtype=all_labels[0].dtype) + np.nan
    maskv = np.reshape(np.concatenate((maskv,nans),axis=1),(-1,))
    maskidx = np.nonzero(maskv)[0]
  naxr = int(np.ceil(nfeat/naxc))
  if ax is None:
    fig,ax = plt.subplots(naxr,naxc,sharex='all',figsize=(20,20))
    ax = ax.flatten()
    plt.tight_layout(h_pad=0)

  pred_cmap = lambda x: plt.get_cmap("tab10")(x%10)
  discreteidx = list(all_labels[0].idx_multidiscrete_to_multi)
  outnames = all_labels[0].get_multi_names()
  for i,feati in enumerate(featidxplot):
    ax[i].cla()
    ti = ax[i].set_title(outnames[feati],y=1.0,pad=-14,color=pred_cmap(feati),loc='left')
    
    if feati in discreteidx:
      disci = discreteidx.index(feati)
      predcurr = predv_discrete[:,disci,:].T
      labelscurr = labelsv_discrete[:,disci,:].T
      zlabels = np.nanmax(labelscurr)
      zpred = np.nanmax(predcurr)
      im = np.stack((1-labelscurr/zlabels,1-predcurr/zpred,np.ones(predcurr.shape)),axis=-1)
      im[np.isnan(im)] = 1.
      ax[i].imshow(im,aspect='auto')
    else:    
      lims = np.nanpercentile(np.concatenate([labelsv_cont[:,feati],predv_cont[:,feati]],axis=0),[prctile_lim,100-prctile_lim])
      ax[i].plot(labelsv_cont[:,feati],'k-',label='True')
      if ismasked:
        ax[i].plot(maskidx,predv_cont[maskidx,i],'.',color=pred_cmap(feati),label='Pred')
      else:
        ax[i].plot(predv_cont[:,feati],'-',color=pred_cmap(feati),label='Pred')
      #ax[i].set_ylim([-ylim_nstd,ylim_nstd])
      ax[i].set_ylim(lims)
      if outnames is not None:
        plt.setp(ti,color=pred_cmap(i))
  ax[0].set_xlim([0,labelsv_cont.shape[0]])
    
  return fig,ax



def animate_pose(Xkps,focusflies=[],ax=None,fig=None,t0=0,
                 figsizebase=11,ms=6,lw=1,focus_ms=12,focus_lw=3,
                 titletexts={},savevidfile=None,fps=30,trel0=0,
                 inputs=None,nstd_input=3,contextl=10,axinput=None,
                 attn_weights=None,skeledgecolors=None,
                 globalpos_future=None,tspred_future=None,
                 futurecolor=[0,0,0,.25],futurelw=1,futurems=6,
                 futurealpha=.25):
  
  #ani = animate_pose(Xkps,focusflies=focusflies,t0=t0,titletexts=titletexts,trel0=np.maximum(0,config['contextl']-64),
  #                  inputs=inputs,contextl=config['contextl']-1,attn_weights=attn_weights,
  #                  globalpos_future={'Pred': globalposfuture},
  #                  tspred_future=dataset.tspred_global)

  plotinput = inputs is not None and len(inputs) > 0

  # attn_weights[key] should be T x >=contextl x nfocusflies
  plotattn = attn_weights is not None
  
  plotfuture = globalpos_future is not None

  ninputs = 0
  if plotinput:
    inputnames = []
    for v in inputs.values():
      if v is not None:
        inputnames = list(v.keys())
        break
    ninputs = len(inputnames)
    if ninputs == 0:
      plotinput = False
      
  if plotinput or plotattn:
    naxc = len(Xkps)
    naxr = 1
    nax = naxc*naxr
  else:
    nax = len(Xkps)
    naxc = int(np.ceil(np.sqrt(nax)))
    naxr = int(np.ceil(nax/naxc))
  
  if plotattn:
    nsubax = ninputs + 1
  else:
    nsubax = ninputs
  
  # get rid of blank flies
  Xkp = list(Xkps.values())[0]
  T = Xkp.shape[-2]
  isreal = mabe.get_real_flies(Xkp)
  nflies = Xkp.shape[-1]
  isfocusfly = np.zeros(nflies,dtype=bool)
  isfocusfly[focusflies] = True
  for Xkp in Xkps.values():
    assert(nflies == Xkp.shape[-1])
    isreal = isreal | mabe.get_real_flies(Xkp)

  for k,v in Xkps.items():
    Xkps[k] = v[...,isreal]
  focusflies = np.nonzero(isfocusfly[isreal])[0]
    
  nflies = np.count_nonzero(isreal)

  minv = -mabe.ARENA_RADIUS_MM*1.01
  maxv = mabe.ARENA_RADIUS_MM*1.01
  
  h = {}

  trel = trel0
  t = t0+trel
  createdax = False
  if ax is None:
    if fig is None:
      fig = plt.figure()
      if plotinput or plotattn:
        fig.set_figheight(figsizebase*1.5)
      else:
        fig.set_figheight(figsizebase*naxr)
      fig.set_figwidth(figsizebase*naxc)

    if plotinput or plotattn:
      gs = matplotlib.gridspec.GridSpec(3,len(Xkps)*nsubax, figure=fig)
      ax = np.array([fig.add_subplot(gs[:2,nsubax*i:nsubax*(i+1)]) for i in range(len(Xkps))])
    else:
      ax = fig.subplots(naxr,naxc)

    for axcurr in ax:
      axcurr.set_xticks([])      
      axcurr.set_yticks([])
    createdax = True
  else:
    assert(ax.size>=nax)
  ax = ax.flatten()
  if (plotinput or plotattn) and (axinput is None):
    gs = matplotlib.gridspec.GridSpec(3,len(Xkps)*nsubax, figure=fig)
    axinput = {}
    for i,k in enumerate(Xkps):
      if k in inputs:
        axinput[k] = np.array([fig.add_subplot(gs[-1,i*nsubax+j]) for j in range(nsubax)])
        for axcurr in axinput[k][1:]:
          axcurr.set_yticks([])

    createdax = True

  if createdax:
    fig.tight_layout()

  h['kpt'] = []
  h['edge'] = []
  h['ti'] = []
  if plotfuture:
    h['future'] = []
    nsamples = {k: globalpos_future[k].shape[0] for k in globalpos_future.keys()}
  
  titletext_ts = np.array(list(titletexts.keys()))
  
  if 0 in titletexts:
    titletext_str = titletexts[0]
  else:
    titletext_str = ''

  for i,k in enumerate(Xkps):
    
    if plotfuture and k in globalpos_future:
      hfuture = []
      ntsfuture = globalpos_future[k].shape[2]
      for j in range(len(focusflies)):
        futurecolors = plt.get_cmap('jet')(np.linspace(0,1,ntsfuture))
        futurecolors[:,-1] = futurealpha
        hfuturefly = [None,]*ntsfuture
        for tfuturei in range(ntsfuture-1,-1,-1):
          hfuturecurr = ax[i].plot(globalpos_future[k][:,trel,tfuturei,0,j],globalpos_future[k][:,trel,tfuturei,1,j],'.',
                                   color=futurecolors[tfuturei],ms=futurems,lw=futurelw)[0]
          hfuturefly[tfuturei] = hfuturecurr
        # for samplei in range(nsamples[k]):
        #   hfuturecurr = ax[i].plot(globalpos_future[k][samplei,trel,:,0,j],globalpos_future[k][samplei,trel,:,1,j],'.-',color=futurecolor,ms=futurems,lw=futurelw)[0]
        #   hfuturefly.append(hfuturecurr)
        hfuture.append(hfuturefly)
      h['future'].append(hfuture)
    
    hkpt,hedge,_,_,_ = mabe.plot_flies(Xkps[k][...,trel,:],ax=ax[i],kpt_ms=ms,skel_lw=lw,skeledgecolors='tab20')

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

  if plotinput or plotattn:
    h['input'] = {}
    t0input = np.maximum(0,trel-contextl)
    contextlcurr = trel0-t0input+1

    if plotinput:
      for k in inputs.keys():
        h['input'][k] = []
        for i,inputname in enumerate(inputnames):
          inputcurr = inputs[k][inputname][trel+1:t0input:-1,:]
          if contextlcurr < contextl:
            pad = np.zeros([contextl-contextlcurr,]+list(inputcurr.shape)[1:])
            pad[:] = np.nan
            inputcurr = np.r_[inputcurr,pad]
          hin = axinput[k][i].imshow(inputcurr,vmin=-nstd_input,vmax=nstd_input,cmap='coolwarm')
          axinput[k][i].set_title(inputname)
          axinput[k][i].axis('auto')
          h['input'][k].append(hin)
    if plotattn:
      for k in attn_weights.keys():
        if k not in h['input']:
          h['input'][k] = []
        # currently only support one focus fly
        hattn = axinput[k][-1].plot(attn_weights[k][trel,-contextl:,0],np.arange(contextl,0,-1))[0]
        #axinput[k][-1].set_xscale('log')
        axinput[k][-1].set_ylim([-.5,contextl-.5])
        axinput[k][-1].set_xlim([0,1])
        axinput[k][-1].invert_yaxis()
        axinput[k][-1].set_title('attention')
        h['input'][k].append(hattn)

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
        if plotfuture and k in globalpos_future:
          ntsfuture = globalpos_future[k].shape[2]
          for j in range(len(focusflies)):
            
            for tfuturei in range(ntsfuture-1,-1,-1):
              h['future'][i][j][tfuturei].set_xdata(globalpos_future[k][:,trel,tfuturei,0,j])
              h['future'][i][j][tfuturei].set_ydata(globalpos_future[k][:,trel,tfuturei,1,j])
            
            # for samplei in range(nsamples[k]):
            #   h['future'][i][j][samplei].set_xdata(globalpos_future[k][samplei,trel,:,0,j])
            #   h['future'][i][j][samplei].set_ydata(globalpos_future[k][samplei,trel,:,1,j])
        if i == 0:
          h['ti'][i].set_text(f'{titletext_str} {k}, t = {t}')
        else:
          h['ti'][i].set_text(k)

      if plotinput or plotattn:
        t0input = np.maximum(0,trel-contextl)
        contextlcurr = trel-t0input+1
      
      if plotinput:
        for k in inputs.keys():
          for i,inputname in enumerate(inputnames):
            
            inputcurr = inputs[k][inputname][trel+1:t0input:-1,:]
            if contextlcurr < contextl:
              pad = np.zeros([contextl-contextlcurr,]+list(inputcurr.shape)[1:])
              pad[:] = np.nan
              inputcurr = np.r_[inputcurr,pad]
            h['input'][k][i].set_data(inputcurr)
            
      if plotattn:
        for k in attn_weights.keys():
          attn_curr = attn_weights[k][trel,-contextl:,0]
          h['input'][k][-1].set_xdata(attn_curr)
          # if any(np.isnan(attn_curr)==False):
          #   axinput[k][-1].set_xlim([0,np.nanmax(attn_curr)])
      return hlist
    
  ani = animation.FuncAnimation(fig, update, frames=range(trel0,T))

  if savevidfile is not None:
    print('Saving animation to file %s...'%savevidfile)
    writer = animation.PillowWriter(fps=30)
    ani.save(savevidfile,writer=writer)
    print('Finished writing.')

  return ani

def json_load_helper(jsonfile):
  with open(jsonfile,'r') as f:
    config = json.load(f)
  config = {k: v for k,v in config.items() if re.search('^_comment',k) is None}
  return config

def read_config(jsonfile):
  
  config = json_load_helper(DEFAULTCONFIGFILE)
  config1 = json_load_helper(jsonfile)  

  # destructive to config
  overwrite_config(config,config1)
  
  config['intrainfile'] = os.path.join(config['datadir'],config['intrainfilestr'])
  config['invalfile'] = os.path.join(config['datadir'],config['invalfilestr'])
  
  if type(config['flatten_obs_idx']) == str:
    if config['flatten_obs_idx'] == 'sensory':
      config['flatten_obs_idx'] = get_sensory_feature_idx()
    else:
      raise ValueError(f"Unknown type {config['flatten_obs_idx']} for flatten_obs_idx")

  # discreteidx will reference mabe.posenames
  if type(config['discreteidx']) == str:
    if config['discreteidx'] == 'global':
      config['discreteidx'] = featglobal.copy()
    else:
      raise ValueError(f"Unknown type {config['discreteidx']} for discreteidx")
  if type(config['discreteidx']) == list:
    for i,v in enumerate(config['discreteidx']):
      if type(v) == str:
        config['discreteidx'][i] = mabe.posenames.index(v)
    config['discreteidx'] = np.array(config['discreteidx'])
    
  if config['modelstatetype'] == 'prob' and config['minstateprob'] is None:
    config['minstateprob'] = 1/config['nstates']
    
  if 'all_discretize_epsilon' in config:
    config['all_discretize_epsilon'] = np.array(config['all_discretize_epsilon'])
    if 'discreteidx' in config and config['discreteidx'] is not None:
      config['discretize_epsilon'] = config['all_discretize_epsilon'][config['discreteidx']]
    
  if 'input_noise_sigma' in config and config['input_noise_sigma'] is not None:
    config['input_noise_sigma'] = np.array(config['input_noise_sigma'])
  #elif 'input_noise_sigma_mult' in config and 'all_discretize_epsilon' in config:
  #  config['input_noise_sigma'] = np.zeros(config['all_discretize_epsilon'].shape)
  #  l = len(config['input_noise_sigma_mult'])
  #  config['input_noise_sigma'][:l] = config['all_discretize_epsilon'][:l]*np.array(config['input_noise_sigma_mult'])    
    
  assert config['modeltype'] in ['mlm','clm']
  assert config['modelstatetype'] in ['prob','best',None]
  assert config['masktype'] in ['ind','block',None]
  
  if ('obs_embedding_types' in config) and (type(config['obs_embedding_types']) == dict):
    for k,v in config['obs_embedding_types'].items():
      if v == 'conv1d':
        # modernize
        config['obs_embedding_types'][k] = 'conv1d_feat'
    if 'obs_embedding_params' not in config:
      config['obs_embedding_params'] = {}
    else:
      if type(config['obs_embedding_params']) != dict:
        assert config['obs_embedding_params'] is None
        config['obs_embedding_params'] = {}

    for k,et in config['obs_embedding_types'].items():
      if k not in config['obs_embedding_params']:
        config['obs_embedding_params'][k] = {}
      params = config['obs_embedding_params'][k]
      if et == 'conv1d_time':
        if 'stride' not in params:
          params['stride'] = 1
        if 'dilation' not in params:
          params['dilation'] = 1
        if 'kernel_size' not in params:
          params['kernel_size'] = 2
        if 'padding' not in params:
          w = (params['stride']-1)+(params['kernel_size']*params['dilation'])-1
          params['padding'] = (w,0)
        if 'channels' not in params:
          params['channels'] = [64,256,512]
      elif et == 'conv2d':
        if 'stride' not in params:
          params['stride'] = (1,1)
        elif type(params['stride']) == int:
          params['stride'] = (params['stride'],params['stride'])
        if 'dilation' not in params:
          params['dilation'] = (1,1)
        elif type(params['dilation']) == int:
          params['dilation'] = (params['dilation'],params['dilation'])
        if 'kernel_size' not in params:
          params['kernel_size'] = (2,3)
        elif type(params['kernel_size']) == int:
          params['kernel_size'] = (params['kernel_size'],params['kernel_size'])
        if 'padding' not in params:
          w1 = (params['stride'][0]-1)+(params['kernel_size'][0]*params['dilation'][0])-1
          w2 = (params['stride'][1]-1)+(params['kernel_size'][1]*params['dilation'][1])-1
          w2a = int(np.ceil(w2/2))
          params['padding'] = ((w1,0),(w2a,w2-w2a))
          #params['padding'] = 'same'
        if 'channels' not in params:
          params['channels'] = [16,64,128]
      elif et == 'conv1d_feat':
        if 'stride' not in params:
          params['stride'] = 1
        if 'dilation' not in params:
          params['dilation'] = 1
        if 'kernel_size' not in params:
          params['kernel_size'] = 3
        if 'padding' not in params:
          params['padding'] = 'same'
        if 'channels' not in params:
          params['channels'] = [16,64,128]
      elif et == 'fc':
        pass
      else:
        raise ValueError(f'Unknown embedding type {et}')
      # end switch over embedding types
    # end if obs_embedding_types in config
  
  return config
    
def overwrite_config(config0,config1,no_overwrite=[]):
  # maybe fix: no_overwrite is just a list of parameter names. this may fail in recursive calls
  for k,v in config1.items():
    if k in no_overwrite:
      continue
    if (k in config0) and (config0[k] is not None) and (type(v) == dict):
      overwrite_config(config0[k],config1[k],no_overwrite=no_overwrite)
    else:
      config0[k] = v
  return

def load_and_filter_data(infile,config):
  
  print(f"loading raw data from {infile}...")
  data = load_raw_npz_data(infile)

  if (len(config['discreteidx']) > 0) and config['discretize_epsilon'] is None:
    if (config['all_discretize_epsilon'] is None):
      scale_perfly = compute_scale_allflies(data)
      config['all_discretize_epsilon'] = compute_noise_params(data,scale_perfly,simplify_out=config['simplify_out'])
    config['discretize_epsilon'] = config['all_discretize_epsilon'][config['discreteidx']]

  # filter out data
  print('filtering data...')
  if config['categories'] is not None and len(config['categories']) > 0:
    filter_data_by_categories(data,config['categories'])
    
  # augment by flipping
  if 'augment_flip' in config and config['augment_flip']:
    flipvideoidx = np.max(data['videoidx'])+1+data['videoidx']
    data['videoidx'] = np.concatenate((data['videoidx'],flipvideoidx),axis=0)
    firstid = np.max(data['ids'])+1
    flipids = data['ids'].copy()
    flipids[flipids>=0] += firstid
    data['ids'] = np.concatenate((data['ids'],flipids),axis=0)
    data['frames'] = np.tile(data['frames'],(2,1))
    flipX = mabe.flip_flies(data['X'])
    data['X'] = np.concatenate((data['X'],flipX),axis=2)
    data['y'] = np.tile(data['y'],(1,2,1))
    data['isdata'] = np.tile(data['isdata'],(2,1))
    data['isstart'] = np.tile(data['isstart'],(2,1))

  # compute scale parameters
  print('computing scale parameters...')
  scale_perfly = compute_scale_allflies(data)

  if np.isnan(SENSORY_PARAMS['otherflies_touch_mult']):
    print('computing touch parameters...')
    SENSORY_PARAMS['otherflies_touch_mult'] = compute_otherflies_touch_mult(data)

  # throw out data that is missing scale information - not so many frames
  idsremove = np.nonzero(np.any(np.isnan(scale_perfly),axis=0))[0]
  data['isdata'][np.isin(data['ids'],idsremove)] = False

  return data,scale_perfly

def criterion_wrapper(example,pred,criterion,dataset,config):
  tgt_continuous,tgt_discrete = dataset.get_continuous_discrete_labels(example)
  pred_continuous,pred_discrete = dataset.get_continuous_discrete_labels(pred)
  tgt = {'labels': tgt_continuous, 'labels_discrete': tgt_discrete}
  pred1 = {'continuous': pred_continuous, 'discrete': pred_discrete}
  if config['modeltype'] == 'mlm':
    if dataset.discretize:
      loss,loss_discrete,loss_continuous = criterion(tgt,pred1,mask=example['mask'].to(pred.device),
                                                     weight_discrete=config['weight_discrete'],extraout=True)
    else:
      loss = criterion(tgt_continuous.to(device=pred.device),pred_continuous,
                      example['mask'].to(pred.device))
      loss_continuous = loss
      loss_discrete = 0.
  else:
    if dataset.discretize:
      loss,loss_discrete,loss_continuous = criterion(tgt,pred1,weight_discrete=config['weight_discrete'],extraout=True)
    else:
      loss = criterion(tgt_continuous.to(device=pred.device),pred_continuous)
      loss_continuous = loss
      loss_discrete = 0.
  return loss,loss_discrete,loss_continuous


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
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230228T193725.pth')
# CLM, trained with dropout = 0.8 on movement, more wall touch keypoints
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230302T221828.pth')
# CLM, trained with dropout = 0.8 on movement, other fly touch features
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230303T230750.pth')
# CLM, trained with dropout = 1.0 on movement, other fly touch features, 10 layers, 512 context
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230305T135655.pth')
# CLM with mixed continuous and discrete state
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230419T175759.pth')
# CLM with mixed continuous and discrete state, movement input
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230421T223920.pth')
# flattened CLM, forward, sideways, orientation are binned outputs
#loadmodelfile = os.path.join(savedir,'flyclm_71G01_male_epoch100_20230512T202000.pth')
# flattened CLM, forward, sideways, orientation are binned outputs, do_separate_inputs = True
# loadmodelfile = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/llmnets/flyclm_flattened_mixed_71G01_male_epoch54_20230517T153613.pth'

def get_modeltype_str(config,dataset):
  if config['modelstatetype'] is not None:
    modeltype_str = f"{config['modelstatetype']}_{config['modeltype']}"
  else:
    modeltype_str = config['modeltype']
  if dataset.flatten:
    modeltype_str += '_flattened'
  if dataset.continuous and dataset.discretize:
    reptype = 'mixed'  
  elif dataset.continuous:
    reptype = 'continuous'
  elif dataset.discretize:
    reptype = 'discrete'
  modeltype_str += f'_{reptype}'
  if config['categories'] is None or len(config['categories']) == 0:
    category_str = 'all'
  else:
    category_str = '_'.join(config['categories'])
  modeltype_str += f'_{category_str}'

  return modeltype_str

def initialize_debug_plots(dataset,dataloader,data,name='',tsplot=None,traj_nsamplesplot=3):

  example_batch = next(iter(dataloader))
  example = FlyExample(example_in=example_batch,dataset=dataset)

  # plot to visualize input features
  fig,ax = debug_plot_sample(example,dataset)

  # plot to check that we can get poses from examples
  hpose,ax,fig = debug_plot_pose(example,dataset,data=data,tsplot=tsplot)
  ax[-1,0].set_xlabel('Train')

  # plot to visualize motion outputs
  axtraj,figtraj = debug_plot_batch_traj(example,dataset,data=data,
                                         label_true='Label',
                                         label_pred='Raw',
                                         nsamplesplot=traj_nsamplesplot)
  figtraj.set_figheight(18)
  figtraj.set_figwidth(12)
  axtraj[0].set_title(name)
  figtraj.tight_layout()
  
  hdebug = {
    'figpose': fig,
    'axpose': ax,
    'hpose': hpose,
    'figtraj': figtraj,
    'axtraj': axtraj,
    'hstate': None,
    'axstate': None,
    'figstate': None,
    'example': example
  }

  plt.show()
  plt.pause(.001)
  
  return hdebug
  
def update_debug_plots(hdebug,config,model,dataset,example,pred,criterion=None,name='',tsplot=None,traj_nsamplesplot=3):
  
  if config['modelstatetype'] == 'prob':
    pred1 = model.maxpred({k: v.detach() for k,v in pred.items()})
  elif config['modelstatetype'] == 'best':
    pred1 = model.randpred(pred.detach())
  else:
    if isinstance(pred,dict):
      pred1 = {k: v.detach().cpu() for k,v in pred.items()}
    else:
      pred1 = pred.detach().cpu()
  debug_plot_pose(example,dataset,pred=pred1,h=hdebug['hpose'],ax=hdebug['axpose'],fig=hdebug['figpose'],tsplot=tsplot)
  debug_plot_batch_traj(example,dataset,criterion=criterion,config=config,pred=pred1,ax=hdebug['axtraj'],fig=hdebug['figtraj'],nsamplesplot=traj_nsamplesplot)
  if config['modelstatetype'] == 'prob':
    hstate,axstate,figstate = debug_plot_batch_state(pred['stateprob'].detach().cpu(),nsamplesplot=3,
                                                      h=hdebug['hstate'],ax=hdebug['axstate'],fig=hdebug['figstate'])
    hdebug['axstate'][0].set_title(name)

  hdebug['axtraj'][0].set_title(name)
  
def initialize_loss_plots(loss_epoch):

  nax = len(loss_epoch)//2
  assert (nax >= 1) and (nax <= 3)
  hloss = {}
    
  hloss['fig'],hloss['ax'] = plt.subplots(nax,1)
  if nax == 1:
    hloss['ax'] = [hloss['ax'],]
    
  hloss['train'], = hloss['ax'][0].plot(loss_epoch['train'].cpu(),'.-',label='Train')
  hloss['val'], = hloss['ax'][0].plot(loss_epoch['val'].cpu(),'.-',label='Val')
  
  if 'train_continuous' in loss_epoch:
    hloss['train_continuous'], = hloss['ax'][1].plot(loss_epoch['train_continuous'].cpu(),'.-',label='Train continuous')
  if 'train_discrete' in loss_epoch:
    hloss['train_discrete'], = hloss['ax'][2].plot(loss_epoch['train_discrete'].cpu(),'.-',label='Train discrete')
  if 'val_continuous' in loss_epoch:
    hloss['val_continuous'], = hloss['ax'][1].plot(loss_epoch['val_continuous'].cpu(),'.-',label='Val continuous')
  if 'val_discrete' in loss_epoch:
    hloss['val_discrete'], = hloss['ax'][2].plot(loss_epoch['val_discrete'].cpu(),'.-',label='Val discrete')
        
  hloss['ax'][-1].set_xlabel('Epoch')
  hloss['ax'][-1].set_ylabel('Loss')
  for l in hloss['ax']:
    l.legend()
  return hloss

def update_loss_plots(hloss,loss_epoch):

  hloss['train'].set_ydata(loss_epoch['train'].cpu())
  hloss['val'].set_ydata(loss_epoch['val'].cpu())
  if 'train_continuous' in loss_epoch:
    hloss['train_continuous'].set_ydata(loss_epoch['train_continuous'].cpu())
  if 'train_discrete' in loss_epoch:
    hloss['train_discrete'].set_ydata(loss_epoch['train_discrete'].cpu())
  if 'val_continuous' in loss_epoch:
    hloss['val_continuous'].set_ydata(loss_epoch['val_continuous'].cpu())
  if 'val_discrete' in loss_epoch:
    hloss['val_discrete'].set_ydata(loss_epoch['val_discrete'].cpu())
  for l in hloss['ax']:
    l.relim()
    l.autoscale()

def initialize_loss(train_dataset,config):
  loss_epoch = {}
  keys = ['train','val']
  if train_dataset.discretize:
    keys = keys + ['train_continuous','train_discrete','val_continuous','val_discrete']
  for key in keys:
    loss_epoch[key] = torch.zeros(config['num_train_epochs'])
    loss_epoch[key][:] = torch.nan
  return loss_epoch

def initialize_model(config,train_dataset,device):

  # architecture arguments
  MODEL_ARGS = {
    'd_model': config['d_model'], 
    'nhead': config['nhead'], 
    'd_hid': config['d_hid'], 
    'nlayers': config['nlayers'],
    'dropout': config['dropout']
    }
  if config['modelstatetype'] is not None:
    MODEL_ARGS['nstates'] = config['nstates']
    assert config['obs_embedding'] == False, 'Not implemented'
    assert train_dataset.flatten == False, 'Not implemented'
  if config['modelstatetype'] == 'prob':
    MODEL_ARGS['minstateprob'] = config['minstateprob']
    
  if config['obs_embedding']:
    MODEL_ARGS['input_idx'],MODEL_ARGS['input_szs'] = train_dataset.get_input_shapes()
    MODEL_ARGS['embedding_types'] = config['obs_embedding_types']
    MODEL_ARGS['embedding_params'] = config['obs_embedding_params']
  d_input = train_dataset.d_input
  if train_dataset.flatten:
    MODEL_ARGS['ntokens_per_timepoint'] = train_dataset.ntokens_per_timepoint
    d_input = train_dataset.flatten_dinput
    d_output = train_dataset.flatten_max_doutput
  elif train_dataset.discretize:
    MODEL_ARGS['d_output_discrete'] = train_dataset.d_output_discrete
    MODEL_ARGS['nbins'] = train_dataset.discretize_nbins
    d_output = train_dataset.d_output_continuous
  else:
    d_output = train_dataset.d_output
    
  if config['modelstatetype'] == 'prob':
    model = TransformerStateModel(d_input,d_output,**MODEL_ARGS).to(device)
    criterion = prob_causal_criterion
  elif config['modelstatetype'] == 'min':
    model = TransformerBestStateModel(d_input,d_output,**MODEL_ARGS).to(device)  
    criterion = min_causal_criterion
  else:
    model = TransformerModel(d_input,d_output,**MODEL_ARGS).to(device)
    
    if train_dataset.discretize:
      # this should be maybe be len(train_dataset.discreteidx) / train_dataset.d_output
      config['weight_discrete'] = len(config['discreteidx']) / nfeatures
      if config['modeltype'] == 'mlm':
        criterion = mixed_masked_criterion
      else:
        criterion = mixed_causal_criterion
    else:
      if config['modeltype'] == 'mlm':
        criterion = masked_criterion
      else:
        criterion = causal_criterion
        
  # if train_dataset.dct_m is not None and config['weight_dct_consistency'] > 0:
  #   criterion = lambda tgt,pred,**kwargs: criterion(tgt,pred,**kwargs) + train_dataset.compare_dct_to_next_relative(pred)
          
  return model,criterion

def compute_loss(model,dataloader,dataset,device,mask,criterion,config):
  
  is_causal = dataset.ismasked() == False
  if is_causal:
    mask = None
    
  model.eval()
  with torch.no_grad():
    all_loss = torch.zeros(len(dataloader),device=device)
    loss = torch.tensor(0.0).to(device)
    if dataset.discretize:  
      loss_discrete = torch.tensor(0.0).to(device)
      loss_continuous = torch.tensor(0.0).to(device)
    nmask = 0
    for i,example in enumerate(dataloader):
      pred = model(example['input'].to(device=device),mask=mask,is_causal=is_causal)
      loss_curr,loss_discrete_curr,loss_continuous_curr = criterion_wrapper(example,pred,criterion,dataset,config)

      if config['modeltype'] == 'mlm':
        nmask += torch.count_nonzero(example['mask'])
      else:
        nmask += example['input'].shape[0]*dataset.ntimepoints
      all_loss[i] = loss_curr
      loss+=loss_curr
      if dataset.discretize:
        loss_discrete+=loss_discrete_curr
        loss_continuous+=loss_continuous_curr
      
    loss = loss.item() / nmask

    if dataset.discretize:
      loss_discrete = loss_discrete.item() / nmask
      loss_continuous = loss_continuous.item() / nmask
      return loss,loss_discrete,loss_continuous
    else:
      return loss
    
def predict_all(dataloader,dataset,model,config,mask):
  
  is_causal = dataset.ismasked() == False
  
  with torch.no_grad():
    w = next(iter(model.parameters()))
    device = w.device

  example_params = dataset.get_flyexample_params()
  
  # compute predictions and labels for all validation data using default masking
  all_pred = []
  all_mask = []
  all_labels = []
  # all_pred_discrete = []
  # all_labels_discrete = []
  with torch.no_grad():
    for example in dataloader:
      pred = model.output(example['input'].to(device=device),mask=mask,is_causal=is_causal)
      if config['modelstatetype'] == 'prob':
        pred = model.maxpred(pred)
      elif config['modelstatetype'] == 'best':
        pred = model.randpred(pred)
      if isinstance(pred,dict):
        pred = {k: v.cpu() for k,v in pred.items()}
      else:
        pred = pred.cpu()
      #pred1 = dataset.get_full_pred(pred)
      #labels1 = dataset.get_full_labels(example=example,use_todiscretize=True)
      example_obj = FlyExample(example_in=example,**example_params)
      label_obj = example_obj.labels
      pred_obj = label_obj.copy()
      pred_obj.erase_labels()
      pred_obj.set_prediction(pred)

      for i in range(np.prod(label_obj.pre_sz)):
        all_pred.append(pred_obj.copy_subindex(idx_pre=i))
        all_labels.append(label_obj.copy_subindex(idx_pre=i))
                          
      # if dataset.discretize:
      #   all_pred_discrete.append(pred['discrete'])
      #   all_labels_discrete.append(example['labels_discrete'])
      # if 'mask' in example:
      #   all_mask.append(example['mask'])

  return all_pred,all_labels#,all_mask,all_pred_discrete,all_labels_discrete

def parse_modelfile(modelfile):
  _,filestr = os.path.split(modelfile)
  filestr,_ = os.path.splitext(filestr)
  m = re.match(r'fly(.*)_epoch\d+_(\d{8}T\d{6})',filestr)
  if m is None:
    modeltype_str = ''
    savetime = ''
  else:
    modeltype_str = m.groups(1)[0]
    savetime = m.groups(1)[1]
  return modeltype_str,savetime

def compute_all_attention_weight_rollouts(attn_weights0):
  attn_weights_rollout = None
  firstframe = None
  attn_context = None
  tpred = attn_weights0[0].size #to do check
  for t,w0 in enumerate(attn_weights0):
    if w0 is None: 
      continue
    w = compute_attention_weight_rollout(w0)
    w = w[-1,-1,...]
    if attn_weights_rollout is None:
      attn_weights_rollout = np.zeros((tpred,)+w.shape)
      attn_weights_rollout[:] = np.nan
      firstframe = t
      attn_context = w.shape[0]
    if attn_context < w.shape[0]:
      pad = np.zeros([tpred,w.shape[0]-attn_context,]+list(w.shape)[1:])
      pad[:firstframe,...] = np.nan
      attn_context = w.shape[0]
      attn_weights_rollout = np.concatenate((attn_weights_rollout,pad),axis=1)
    attn_weights_rollout[t,:] = 0.
    attn_weights_rollout[t,:w.shape[0]] = w
  return attn_weights_rollout

def get_pose_future(data,scales,tspred_global,ts=None,fliespred=None):

  maxT = data['X'].shape[2]
  if ts is None:
    ts = np.arange(maxT)
  if fliespred is None:
    fliespred = np.arange(data['X'].shape[3])
  
  Xkpfuture = np.zeros((data['X'].shape[0],data['X'].shape[1],len(ts),len(tspred_global),len(fliespred)))
  Xkpfuture[:] = np.nan
  for ti,toff in enumerate(tspred_global):
    idxcurr = ts<maxT-toff
    tscurr = ts[idxcurr]
    Xkpfuture[:,:,idxcurr,ti] = data['X'][:,:,tscurr+toff][...,fliespred]
    isbad = data['videoidx'][tscurr,0] != data['videoidx'][tscurr+toff,0]
    Xkpfuture[:,:,isbad] = np.nan
  
  relposefuture,globalposfuture = compute_pose_features(Xkpfuture,scales)
  if globalposfuture.ndim == 3: # when there is one fly, it gets collapsed
    globalposfuture = globalposfuture[...,None]
    relposefuture = relposefuture[...,None]
  globalposfuture = globalposfuture.transpose(1,2,0,3)
  relposefuture = relposefuture.transpose(1,2,0,3)
  return globalposfuture,relposefuture

def animate_predict_open_loop(model,dataset,data,scale_perfly,config,fliespred,t0,tpred,burnin=None,debug=False,plotattnweights=False,plotfuture=False,nsamplesfuture=1):
    
  #ani = animate_predict_open_loop(model,val_dataset,valdata,val_scale_perfly,config,fliespred,t0,tpred,debug=False,
  #                            plotattnweights=False,plotfuture=train_dataset.ntspred_global>1,nsamplesfuture=nsamplesfuture)
    
  if burnin is None:
    burnin = config['contextl']-1

  Xkp_true = data['X'][...,t0:t0+tpred+dataset.ntspred_max,:].copy()
  Xkp = Xkp_true.copy()

  ids = data['ids'][t0,fliespred]
  scales = scale_perfly[:,ids]


  #fliespred = np.nonzero(mabe.get_real_flies(Xkp))[0]
  for i,flynum in enumerate(fliespred):
    id = data['ids'][t0,flynum]
    scale = scale_perfly[:,id]
    metadata = {'flynum': flynum, 'id': id, 't0': t0, 'videoidx': data['videoidx'][t0,0], 'frame0': data['frames'][t0,0]}
    Xkp_obj = PoseLabels(Xkp=Xkp_true[...,flynum],scale=scale,metadata=metadata,dataset=dataset)


  if plotfuture:
    # subtract one from tspred_global, as the tspred_global for predicted data come from the previous frame
    globalposfuture_true,relposefuture_true = get_pose_future(data,scales,[t+1 for t in dataset.tspred_global],ts=np.arange(t0,t0+tpred),fliespred=fliespred)

  model.eval()

  # capture all outputs of predict_open_loop in a tuple
  res = dataset.predict_open_loop(Xkp,fliespred,scales,burnin,model,maxcontextl=config['contextl'],
                                  debug=debug,need_weights=plotattnweights,nsamples=nsamplesfuture)
  Xkp_pred,zinputs,globalposfuture_pred,relposefuture_pred = res[:4]
  if plotattnweights:
    attn_weights0 = res[4]

  Xkps = {'Pred': Xkp_pred.copy(),'True': Xkp_true.copy()}
  #Xkps = {'Pred': Xkp_pred.copy()}
  if len(fliespred) == 1:
    inputs = {'Pred': split_features(zinputs,axis=1)}
  else:
    inputs = None

  if plotattnweights:
    attn_weights = {'Pred': compute_all_attention_weight_rollouts(attn_weights0)}
  else:
    attn_weights = None

  focusflies = fliespred
  titletexts = {0: 'Initialize', burnin: ''}
  
  if plotfuture:
    future_args = {'globalpos_future': {'Pred': globalposfuture_pred, 'True': globalposfuture_true[None,...]},
                   'tspred_future': dataset.tspred_global}
  else:
    future_args = {}
    
  ani = animate_pose(Xkps,focusflies=focusflies,t0=t0,titletexts=titletexts,trel0=np.maximum(0,config['contextl']-64),
                    inputs=inputs,contextl=config['contextl']-1,attn_weights=attn_weights,
                    **future_args)
  
  return ani

def get_interval_ends(tf):
  tf = np.r_[False,tf,False]
  idxstart = np.nonzero((tf[:-1]==False) & (tf[1:] == True))[0]
  idxend = np.nonzero((tf[:-1]==True) & (tf[1:] == False))[0]
  return idxstart,idxend
  
def split_data_by_id(data):
  splitdata = []
  nflies = data['X'].shape[-1]
  for flynum in range(nflies):
    isdata = data['isdata'][:,flynum] & (data['isstart'][:,flynum]==False)
    idxstart,idxend = get_interval_ends(isdata)
    for i in range(len(idxstart)):
      i0 = idxstart[i]
      i1 = idxend[i]
      id = data['ids'][i0,flynum]
      if data['isdata'][i0-1,flynum] and data['ids'][i0-1,flynum] == id:
        i0 -= 1
      splitdata.append({
        'flynum': flynum,
        'id': id,
        'i0': i0,
        'i1': i1,
      })
  return splitdata
  
def explore_representation(configfile):

  config = read_config(configfile)

  np.random.seed(config['numpy_seed'])
  torch.manual_seed(config['torch_seed'])
  device = torch.device(config['device'])

  plt.ion()
  
  data,scale_perfly = load_and_filter_data(config['intrainfile'],config)
  splitdata = split_data_by_id(data)
  
  
  for i in range(len(splitdata)):
    scurr = splitdata[i]
    fcurr = compute_features(data['X'][...,scurr['i0']:scurr['i1'],:],
                             scurr['id'],scurr['flynum'],scale_perfly,smush=False,simplify_in='no_sensory')
    movecurr = fcurr['labels']
    if i == 0:
      move = movecurr
    else:
      move = np.r_[move,movecurr]

  outnames_global = ['forward','sideways','orientation']
  outnames = outnames_global + [mabe.posenames[x] for x in np.nonzero(featrelative)[0]]

  mu = np.nanmean(move,axis=0)
  sig = np.nanstd(move,axis=0)
  zmove = (move-mu)/sig
  
  # pca = sklearn.decomposition.PCA()
  # pca.fit(zmove)
  
  # fig,ax = plt.subplots(1,1)
  # fig.set_figheight(12)
  # fig.set_figwidth(20)
  # clim = np.max(np.abs(pca.components_))*np.array([-1,1])
  # him = ax.imshow(pca.components_,aspect='auto',clim=clim,cmap='RdBu')
  # fig.colorbar(him,ax=ax)
  # ax.set_xlabel('Movement feature')
  # ax.set_ylabel('PC')
  # ax.set_xticks(np.arange(move.shape[1]))
  # ax.set_xticklabels(outnames)
  # ax.set_yticks(np.arange(pca.components_.shape[0]))
  # ax.tick_params(axis='x', labelrotation = 90)
  # ax.invert_yaxis()
  # ax.set_title('PCA weights')
  # fig.tight_layout()
  
  # ica = sklearn.decomposition.FastICA(whiten='unit-variance')
  # ica.fit(zmove)
  
  # fig,ax = plt.subplots(1,1)
  # fig.set_figheight(12)
  # fig.set_figwidth(20)
  # clim = np.mean(np.max(ica.components_,axis=1),axis=0)*np.array([-1,1])
  # him = ax.imshow(ica.components_,aspect='auto',clim=clim,cmap='RdBu')
  # fig.colorbar(him,ax=ax)
  # ax.set_xlabel('Movement feature')
  # ax.set_ylabel('PC')
  # ax.set_xticks(np.arange(move.shape[1]))
  # ax.set_xticklabels(outnames)
  # ax.set_yticks(np.arange(ica.components_.shape[0]))
  # ax.tick_params(axis='x', labelrotation = 90)
  # ax.invert_yaxis()
  # ax.set_title('ICA weights')
  # fig.tight_layout()
  
  
  # spca = sklearn.decomposition.SparsePCA()
  # spca.fit(zmove)
  
  # fig,ax = plt.subplots(1,1)
  # fig.set_figheight(12)
  # fig.set_figwidth(20)
  # clim = np.max(np.abs(spca.components_))*np.array([-1,1])
  # him = ax.imshow(spca.components_,aspect='auto',clim=clim,cmap='RdBu')
  # fig.colorbar(him,ax=ax)
  # ax.set_xlabel('Movement feature')
  # ax.set_ylabel('PC')
  # ax.set_xticks(np.arange(move.shape[1]))
  # ax.set_xticklabels(outnames)
  # ax.set_yticks(np.arange(spca.components_.shape[0]))
  # ax.tick_params(axis='x', labelrotation = 90)
  # ax.invert_yaxis()
  # ax.set_title('SPCA weights')
  # fig.tight_layout()

  bin_edges = np.zeros((nfeatures,config['discretize_nbins']+1))  
  for feati in range(nfeatures):
    bin_edges[feati,:] = select_bin_edges(move[:,feati],config['discretize_nbins'],config['all_discretize_epsilon'][feati],feati=feati)
  
  featpairs = [
    ['left_front_leg_tip_angle','left_front_leg_tip_dist'],
    ['left_middle_femur_base_angle','left_middle_femur_tibia_joint_angle'],
    ['left_middle_femur_tibia_joint_angle','left_middle_leg_tip_angle'],
    ]
  nax = len(featpairs)
  nc = int(np.ceil(np.sqrt(nax)))
  nr = int(np.ceil(nax/nc))
  fig,ax = plt.subplots(nr,nc,squeeze=False)
  ax = ax.flatten()

  for i in range(len(featpairs)):
    feati = [outnames.index(x) for x in featpairs[i]]
    density,_,_ = np.histogram2d(zmove[:,feati[0]],zmove[:,feati[1]],bins=[bin_edges[feati[0],:],bin_edges[feati[1],:]],density=True)
    ax[i].cla()
    X, Y = np.meshgrid(bin_edges[feati[0],1:-1],bin_edges[feati[1],1:-1])
    density = density[1:-1,1:-1]
    him = ax[i].pcolormesh(X,Y,density, norm=matplotlib.colors.LogNorm(vmin=np.min(density[density>0]), vmax=np.max(density)),edgecolors='k')
    ax[i].set_xlabel(outnames[feati[0]])
    ax[i].set_ylabel(outnames[feati[1]])
  fig.tight_layout()

  # ax[i].plot(move[:,feati[0]],move[:,feati[1]],'.',alpha=.02,markersize=1)
  
  
  valdata,val_scale_perfly = load_and_filter_data(config['invalfile'],config)

def debug_add_noise(train_dataset,data,idxsample=0,tsplot=None):
  # debugging adding noise
  train_dataset.set_eval_mode()
  extrue = train_dataset[idxsample]
  train_dataset.set_train_mode()
  exnoise = train_dataset[idxsample]
  exboth = {}
  for k in exnoise.keys():
    if type(exnoise[k]) == torch.Tensor:
      exboth[k] = torch.stack((extrue[k],exnoise[k]),dim=0)
    elif type(exnoise[k]) == dict:
      exboth[k] = {}
      for k1 in exnoise[k].keys():
        exboth[k][k1] = torch.stack((torch.tensor(extrue[k][k1]),torch.tensor(exnoise[k][k1])))
    else:
      raise ValueError('huh')  
  if tsplot is None:
    tsplot = np.round(np.linspace(0,64,4)).astype(int)
  hpose,ax,fig = debug_plot_pose(exboth,train_dataset,data=data,tsplot=tsplot)
  Xfeat_true = train_dataset.get_Xfeat(example=extrue,use_todiscretize=True)
  Xfeat_noise = train_dataset.get_Xfeat(example=exnoise,use_todiscretize=True)

def clean_intermediate_results(savedir):
  modelfiles = list(pathlib.Path(savedir).glob('*.pth'))
  modelfilenames = [p.name for p in modelfiles]
  p = re.compile('^(?P<prefix>.+)_epoch(?P<epoch>\d+)_(?P<suffix>.*).pth$')
  m = [p.match(n) for n in modelfilenames]
  ids = np.array([x.group('prefix')+'___'+x.group('suffix') for x in m])
  epochs = np.array([int(x.group('epoch')) for x in m])
  uniqueids,idx = np.unique(ids,return_inverse=True)
  removed = []
  nremoved = 0
  for i,id in enumerate(uniqueids):
    idxcurr = np.nonzero(ids==id)[0]
    if len(idxcurr) <= 1:
      continue
    j = idxcurr[np.argmax(epochs[idxcurr])]
    idxremove = idxcurr[epochs[idxcurr]<epochs[j]]
    while True:
      print(f'Keep {modelfilenames[j]} and remove the following files:')
      for k in idxremove:
        print(f'Remove {modelfilenames[k]}')
      r = input('(y/n) ?  ')
      if r == 'y':
        for k in idxremove:
          print(f'Removing {modelfiles[k]}')
          os.remove(modelfiles[k])
          removed.append(modelfiles[k])
          nremoved += 1
        break
      elif r == 'n':
        break
      else:
        print('Bad input, response must be y or n')
  print(f'Removed {nremoved} files')
  return removed

def gzip_pickle_dump(filename, data):
  with gzip.open(filename, 'wb') as f:
    pickle.dump(data, f)

def gzip_pickle_load(filename):
  with gzip.open(filename, 'rb') as f:
    return pickle.load(f)

# import h5py
# def hdf5_save(f, d, name="root"):

#   try:
#     if isinstance(f,str):
#       f = h5py.File(f, "w")
#     if isinstance(d,dict):
#       g = f.create_group('dict__'+name)
#       for k,v in d.items():
#         hdf5_save(g,v,name=k)
#     elif isinstance(d,list):
#       if np.all(np.array([isinstance(x,str) for x in d])):
#         g = f.create_dataset('liststr__'+name,data=np.array(d,dtype='S'))
#       else:
#         g = f.create_group('list__'+name)
#         for i,v in enumerate(d):
#           hdf5_save(g,v,name=str(i))
#     elif isinstance(d,np.ndarray):    
#       if np.all(np.array([isinstance(x,str) for x in d])):
#         g = f.create_dataset('ndarraystr__'+name,data=np.array(d,dtype='S'))
#       else:
#         g = f.create_dataset('ndarray__'+name,data=d)
#     else:
#       g = f.create_dataset('other__'+name,data=d)
  
#   except Exception as e:
#     print(f'Error saving {name}')
#     raise e
  
#   return f

def save_chunked_data(savefile,d):
  gzip_pickle_dump(savefile,d)
  return

def load_chunked_data(savefile):
  return gzip_pickle_load(savefile)

def compute_npad(tspred_global,dct_m):
  npad = np.max(tspred_global)
  if dct_m is not None:
    npad = np.maximum(dct_m.shape[0],npad)
  return npad

def sanity_check_tspred(data,compute_feature_params,npad,scale_perfly,contextl=512,t0=510,flynum=0):
  # sanity check on computing features when predicting many frames into the future
  # compute inputs and outputs for frames t0:t0+contextl+npad+1 with tspred_global set by config
  # and inputs ant outputs for frames t0:t0+contextl+1 with just next frame prediction.
  # the inputs should match each other 
  # the outputs for each of the compute_feature_params['tspred_global'] should match the next frame 
  # predictions for the corresponding frame
  
  epsilon = 1e-6
  id = data['ids'][t0,flynum]

  # compute inputs and outputs with tspred_global = compute_feature_params['tspred_global']
  contextlpad = contextl+npad
  t1 = t0+contextlpad-1
  x = data['X'][...,t0:t1+1,:]
  xcurr1,idxinfo1 = compute_features(x,id,flynum,scale_perfly,outtype=np.float32,returnidx=True,npad=npad,**compute_feature_params)

  # compute inputs and outputs with tspred_global = [1,]
  contextlpad = contextl+1
  t1 = t0+contextlpad-1
  x = data['X'][...,t0:t1+1,:]
  xcurr0,idxinfo0 = compute_features(x,id,flynum,scale_perfly,outtype=np.float32,tspred_global=[1,],returnidx=True,
                                     **{k: v for k,v in compute_feature_params.items() if k != 'tspred_global'})

  assert np.all(np.abs(xcurr0['input']-xcurr1['input']) < epsilon)
  for f in featglobal:
    # find row of np.array idxinfo1['labels']['global_feat_tau'] that equals (f,1)
    i1 = np.nonzero((idxinfo1['labels']['global_feat_tau'][:,0] == f) & (idxinfo1['labels']['global_feat_tau'][:,1] == 1))[0][0]
    i0 = np.nonzero((idxinfo0['labels']['global_feat_tau'][:,0] == f) & (idxinfo0['labels']['global_feat_tau'][:,1] == 1))[0][0]
    assert np.all(np.abs(xcurr1['labels'][:,i1]-xcurr0['labels'][:,i0])<epsilon)

  return

def sanity_check_temporal_dep(train_dataloader,device,train_src_mask,is_causal,model,tmess=300):
  # sanity check on temporal dependences
  # create xin2 that is like xin, except xin2 from frame tmess onwards is set to garbage value 100.
  # make sure that model(xin) and model(xin2) matches before frame tmess
  x = next(iter(train_dataloader))
  xin = x['input'].clone()
  xin2 = xin.clone()
  tmess = 300
  xin2[:,tmess:,:] = 100.
  model.eval()
  with torch.no_grad():
    pred = model(xin.to(device),mask=train_src_mask,is_causal=is_causal)
    pred2 = model(xin2.to(device),mask=train_src_mask,is_causal=is_causal)
  if type(pred) == dict:
    for k in pred.keys():
      matches = torch.all(pred2[k][:,:tmess]==pred[k][:,:tmess]).item()
      assert matches
  else:
    matches = torch.all(pred2[:,:tmess]==pred[:,:tmess]).item()
    assert matches
    
def compare_dicts(old_ex,new_ex,maxerr=None):
  for k,v in old_ex.items():
    err = 0.
    if not k in new_ex:
      print(f'Missing key {k}')
    elif type(v) is torch.Tensor:
      v = v.cpu().numpy()
      newv = new_ex[k]
      if type(newv) is torch.Tensor:
        newv = newv.cpu().numpy()
      err = np.nanmax(np.abs(v-newv))
      print(f'max diff {k}: {err:e}')
    elif type(v) is np.ndarray:
      err = np.nanmax(np.abs(v-new_ex[k]))
      print(f'max diff {k}: {err:e}')
    elif type(v) is dict:
      print(f'Comparing dict {k}')
      compare_dicts(v,new_ex[k])
    else:
      try:
        err = np.nanmax(np.abs(v-new_ex[k]))
        print(f'max diff {k}: {err:e}')
      except:
        print(f'not comparing {k}')
    if maxerr is not None:
      assert err < maxerr

  return

def data_to_kp_from_metadata(data,metadata,ntimepoints):
  t0 = metadata['t0']
  flynum = metadata['flynum']
  id = metadata['id']
  datakp = data['X'][:,:,t0:t0+ntimepoints+1,flynum].transpose(2,0,1)
  return datakp,id

def debug_less_data(data,T=10000):
  data['videoidx'] = data['videoidx'][:T,:]
  data['ids'] = data['ids'][:T,:]
  data['frames'] = data['frames'][:T,:]
  data['X'] = data['X'][:,:,:T,:]
  data['y'] = data['y'][:,:T,:]
  data['isdata'] = data['isdata'][:T,:]
  data['isstart'] = data['isstart'][:T,:]
  return

def debug_fly_example(configfile=None,loadmodelfile=None,restartmodelfile=None):
  #tmpsavefile = 'chunkeddata20240325.pkl'
  #tmpsavefile = 'chunkeddata20240325_decimated.pkl'
  tmpsavefile = 'tmp_small_usertrainval.pkl'

  configfile = 'config_fly_llm_multitime_20230825.json'

  # configuration parameters for this model
  config = read_config(configfile)

  # debug velocity representation
  config['compute_pose_vel'] = True

  # debug dct
  config['dct_tau'] = 4
  
  # debug no multi time-scale predictions
  #config['tspred_global'] = [1,]
  #config['discrete_tspred'] = [1,]

  if os.path.exists(tmpsavefile):
    with open(tmpsavefile,'rb') as f:
      tmp = pickle.load(f)
      data = tmp['data']
      scale_perfly = tmp['scale_perfly']
  else:
    data,scale_perfly = load_and_filter_data(config['intrainfile'],config)
    valdata,val_scale_perfly = load_and_filter_data(config['invalfile'],config)
    T = 10000
    debug_less_data(data,T)
    debug_less_data(valdata,T)
    
    with open(tmpsavefile,'wb') as f:
      pickle.dump({'data': data, 'scale_perfly': scale_perfly, 'valdata': valdata, 'val_scale_perfly': val_scale_perfly},f)

  if config['dct_tau'] is not None and config['dct_tau'] > 0:
    dct_m,idct_m = get_dct_matrix(config['dct_tau'])
  else:
    dct_m = None
    idct_m = None
  
  # how much to pad outputs by -- depends on how many frames into the future we will predict
  npad = compute_npad(config['tspred_global'],dct_m)
  chunk_data_params = {'npad': npad}
  
  compute_feature_params = {
     "simplify_out": config['simplify_out'],
     "simplify_in": config['simplify_in'],
     "dct_m": dct_m,
     "tspred_global": config['tspred_global'],
     "compute_pose_vel": config['compute_pose_vel'],
     "discreteidx": config['discreteidx'],
  }

  # function for computing features
  reparamfun = lambda x,id,flynum,**kwargs: compute_features(x,id,flynum,scale_perfly,outtype=np.float32,
                                                            **compute_feature_params,**kwargs)
  # chunk the data if we didn't load the pre-chunked cache file
  print('Chunking training data...')
  X = chunk_data(data,config['contextl'],reparamfun,**chunk_data_params)

  dataset_params = {
    'max_mask_length': config['max_mask_length'],
    'pmask': config['pmask'],
    'masktype': config['masktype'],
    'simplify_out': config['simplify_out'],
    'pdropout_past': config['pdropout_past'],
    'input_labels': config['input_labels'],
    'dozscore': True,
    'discreteidx': config['discreteidx'],
    'discretize_nbins': config['discretize_nbins'],
    'discretize_epsilon': config['discretize_epsilon'],
    'flatten_labels': config['flatten_labels'],
    'flatten_obs_idx': config['flatten_obs_idx'],
    'flatten_do_separate_inputs': config['flatten_do_separate_inputs'],
    'p_add_input_noise': config['p_add_input_noise'],
    'dct_ms': (dct_m,idct_m),
    'tspred_global': config['tspred_global'],
    'discrete_tspred': config['discrete_tspred'],
    'compute_pose_vel': config['compute_pose_vel'],
  }
  train_dataset_params = {
    'input_noise_sigma': config['input_noise_sigma'],
  }

  compute_feature_params = {
     "simplify_out": config['simplify_out'],
     "simplify_in": config['simplify_in'],
     "dct_m": dct_m,
     "tspred_global": config['tspred_global'],
     "compute_pose_vel": config['compute_pose_vel'],
     "discreteidx": config['discreteidx'],
  }

  # import old_fly_llm
  
  # old_dataset_params = dataset_params.copy()
  # old_dataset_params.pop('compute_pose_vel')
  # old_train_dataset = old_fly_llm.FlyMLMDataset(X,**train_dataset_params,**old_dataset_params)
  # train_dataloader = torch.utils.data.DataLoader(old_train_dataset,
  #                                               batch_size=1,
  #                                               shuffle=False,
  #                                               pin_memory=True,
  #                                               )

  # old_batch = next(iter(train_dataloader))
  # old_fly_llm.debug_plot_batch_pose(old_batch,old_train_dataset)

  print('Creating training data set...')
  train_dataset = FlyMLMDataset(X,**train_dataset_params,**dataset_params)
  
  flyexample = train_dataset.data[0]
  
  # compare flyexample initialized from FlyMLMDataset and from keypoints directly
  contextlpad = train_dataset.contextl + npad + 1
  Xkp = data['X'][:,:,flyexample.metadata['t0']:flyexample.metadata['t0']+contextlpad,:]
  flyexample_kp = FlyExample(Xkp=Xkp,scale=scale_perfly[:,flyexample.metadata['id']],
                             flynum=flyexample.metadata['flynum'],metadata=flyexample.metadata,
                             **train_dataset.get_flyexample_params())
  print(f"comparing flyexample initialized from FlyMLMDataset and from keypoints directly")
  err = np.max(np.abs(flyexample_kp.labels.labels_raw['continuous']-flyexample.labels.labels_raw['continuous']))
  print('max diff between continuous labels: %e'%err)
  assert err < 1e-9
  err = np.max(np.abs(flyexample_kp.labels.labels_raw['discrete']-flyexample.labels.labels_raw['discrete'])) 
  print('max diff between discrete labels: %e'%err)
  assert err < 1e-9
  err = np.max(np.abs(flyexample_kp.labels.labels_raw['todiscretize']-flyexample.labels.labels_raw['todiscretize']))
  print('max diff between todiscretize: %e'%err)  
  assert err < 1e-9
  multi = flyexample.labels.get_multi(use_todiscretize=True,zscored=False)
  multi_kp = flyexample_kp.labels.get_multi(use_todiscretize=True,zscored=False)
  err = np.max(np.abs(multi-multi_kp))
  print('max diff between multi labels: %e'%err)
  assert err < 1e-9

  # compare pose feature representation in and outside PoseLabels
  err_chunk_multi = np.max(np.abs(X[0]['labels']-flyexample.labels.get_multi(use_todiscretize=True,zscored=False)))
  print('max diff between chunked labels and multi: %e'%err_chunk_multi)  
  assert err_chunk_multi < 1e-3

  # extract frames associated with metadata in flyexample
  contextl = flyexample.ntimepoints
  datakp,id = data_to_kp_from_metadata(data,flyexample.metadata,contextl)
  # compute next frame pose feature representation directly
  datafeat = mabe.kp2feat(datakp.transpose(1,2,0),scale_perfly[:,id])[...,0].T  

  if config['compute_pose_vel']:

    print('\nComparing next frame movements from train_dataset to those from flyexample')
    chunknext = train_dataset.get_next_movements(movements=X[0]['labels'])
    examplenext = flyexample.labels.get_next(use_todiscretize=True,zscored=False)
    err_chunk_next = np.max(np.abs(chunknext-examplenext))
    print('max diff between chunked labels and next: %e'%err_chunk_next)
    assert err_chunk_next < 1e-3

  else:
    
    print('\nComparing next frame pose feature representation from train_dataset to that from flyexample')
    chunknextcossin = train_dataset.get_next_movements(movements=X[0]['labels'])
    examplenextcossin = flyexample.labels.get_nextcossin(use_todiscretize=True,zscored=False)
    err_chunk_nextcossin = np.max(np.abs(chunknextcossin-examplenextcossin))
    print('max diff between chunked labels and nextcossin: %e'%err_chunk_nextcossin)
    assert err_chunk_nextcossin < 1e-3

    chunknext = train_dataset.convert_cos_sin_to_angle(chunknextcossin)
    examplenext = flyexample.labels.get_next(use_todiscretize=True,zscored=False)
    err_chunk_next = np.max(np.abs(chunknext-examplenext))
    print('max diff between chunked labels and next: %e'%err_chunk_next)
    assert err_chunk_next < 1e-3

    examplefeat = flyexample.labels.get_next_pose(use_todiscretize=True,zscored=False)
    
    err_chunk_data_feat = np.max(np.abs(chunknext[:,featrelative]-datafeat[1:,featrelative]))
    print('max diff between chunked and data relative features: %e'%err_chunk_data_feat)
    assert err_chunk_data_feat < 1e-3

    err_example_chunk_feat = np.max(np.abs(chunknext[:,featrelative]-examplefeat[1:,featrelative]))
    print('max diff between chunked and example relative features: %e'%err_example_chunk_feat)
    assert err_example_chunk_feat < 1e-3

    err_example_data_global = np.max(np.abs(datafeat[:,featglobal]-examplefeat[:,featglobal]))
    print('max diff between data and example global features: %e'%err_example_data_global)
    assert err_example_data_global < 1e-3

    err_example_data_feat = np.max(np.abs(datafeat[:,featrelative]-examplefeat[:,featrelative]))
    print('max diff between data and example relative features: %e'%err_example_data_feat)
    assert err_example_data_feat < 1e-3
  
  examplekp = flyexample.labels.get_next_keypoints(use_todiscretize=True)
  err_mean_example_data_kp = np.mean(np.abs(datakp[:]-examplekp))
  print('mean diff between data and example keypoints: %e'%err_mean_example_data_kp)
  err_max_example_data_kp = np.max(np.abs(datakp[:]-examplekp))
  print('max diff between data and example keypoints: %e'%err_max_example_data_kp)

  debug_plot_pose(flyexample,data=data)  
  # elements of the list tspred_global that are smaller than contextl
  tsplot = [t for t in train_dataset.tspred_global if t < contextl]
  debug_plot_pose(flyexample,pred=flyexample,tsplot=tsplot)
  
  if config['compute_pose_vel']:

    import old_fly_llm
    
    old_dataset_params = dataset_params.copy()
    old_dataset_params.pop('compute_pose_vel')
    old_train_dataset = old_fly_llm.FlyMLMDataset(X,**train_dataset_params,**old_dataset_params,
                                                  discretize_params=train_dataset.get_discretize_params(),
                                                  zscore_params=train_dataset.get_zscore_params())

    new_ex = flyexample.get_train_example()
    old_ex = old_train_dataset[0]
    
    compare_dicts(old_ex,new_ex,maxerr=1e-3)

    print('\nComparing old fly llm code to new:')    
    mean_err_discrete = torch.mean(torch.sqrt(torch.sum((old_ex['labels_discrete']-new_ex['labels_discrete'])**2.,dim=-1))).item()
    print('mean error between old and new discrete labels: %e'%mean_err_discrete)
    assert mean_err_discrete < 1e-3

    max_err_continuous = torch.max(torch.abs(old_ex['labels']-new_ex['labels'])).item()
    print('max error between old and new continuous labels: %e'%max_err_continuous)
    assert max_err_continuous < 1e-3
    
    max_err_input = torch.max(torch.abs(new_ex['input']-old_ex['input'])).item()
    print('max error between old and new input: %e'%max_err_input)
    assert max_err_input < 1e-3
    
    max_err_init = torch.max(torch.abs(new_ex['init']-old_ex['init'])).item()
    print('max error between old and new init: %e'%max_err_init)
    assert max_err_init < 1e-3
  
  # check global future predictions
  print('\nChecking that representations of many frames into the future match')
  for tpred in flyexample.labels.tspred_global:
    examplefuture = flyexample.labels.get_future_globalpos(use_todiscretize=True,tspred=tpred,zscored=False)
    t = flyexample.metadata['t0']+tpred
    flynum = flyexample.metadata['flynum']
    datakpfuture = data['X'][:,:,t:t+contextl,flynum]
    datafeatfuture = mabe.kp2feat(datakpfuture,scale_perfly[:,id])[...,0].T  
    err_global_future = np.max(np.abs(datafeatfuture[:,featglobal]-examplefuture[:,0,:]))
    print(f'max diff between data and t+{tpred} global prediction: {err_global_future:e}')
    assert err_global_future < 1e-3
  
  # check relative future predictions
  if flyexample.labels.ntspred_relative > 1:
    examplefuture = flyexample.labels.get_future_relative_pose(zscored=False,use_todiscretize=True)
    for tpred in range(1,flyexample.labels.ntspred_relative):
      t = flyexample.metadata['t0']+tpred
      datakpfuture = data['X'][:,:,t:t+contextl,flynum]
      datafeatfuture = mabe.kp2feat(datakpfuture,scale_perfly[:,id])[...,0].T  
      err_relative_future = np.max(np.abs(datafeatfuture[:,featrelative]-examplefuture[:,tpred-1,:]))
      print(f'max diff between data and t+{tpred} relative prediction: {err_relative_future:e}')
      assert err_relative_future < 1e-3
      
  # get a training example
  print('\nComparing training example from dataset to creating a new FlyExample from that training example, and converting back to a training example')
  trainexample = train_dataset[0]
  flyexample1 = FlyExample(example_in=trainexample,dataset=train_dataset)
  trainexample1 = flyexample1.get_train_example()
  compare_dicts(trainexample,trainexample1,maxerr=1e-9)
      
  # initialize example from batch
  print('\nComparing training batch to FlyExample created from that batch converted back to a training batch')
  train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=config['batch_size'],
                                                shuffle=False,
                                                pin_memory=True,
                                                )
  raw_batch = next(iter(train_dataloader))
  example_batch = FlyExample(example_in=raw_batch,dataset=train_dataset)
  trainbatch1 = example_batch.get_train_example()
  compare_dicts(raw_batch,trainbatch1,maxerr=1e-9)

  debug_plot_pose(example_batch,data=data)
  debug_plot_sample(example_batch,train_dataset)
  debug_plot_batch_traj(example_batch,train_dataset)
  #old_fly_llm.debug_plot_sample_inputs(old_train_dataset,raw_batch)
  
  print('Goodbye!')
  plt.show(block=True)

def main(configfile,loadmodelfile=None,restartmodelfile=None):

  tmpsavefile = ''
  # to save time, i saved the chunked data to a pkl file
  #tmpsavefile = 'chunkeddata20230905.pkl'
  #tmpsavefile = 'chunkeddata20230828.pkl'
  #tmpsavefile = 'chunkeddata20240325.pkl'
  doloadtmpsavefile = os.path.exists(tmpsavefile)
  #tmpsavefile = None
  debugdatafile = ''
  debugdatafile = 'tmp_small_usertrainval.pkl'
  isdebug = len(debugdatafile) > 0

  # configuration parameters for this model
  config = read_config(configfile)
  
  # set loadmodelfile and restartmodelfile from config if not specified
  if loadmodelfile is None and 'loadmodelfile' in config:
    loadmodelfile = config['loadmodelfile']
  if restartmodelfile is None and 'restartmodelfile' in config:
    loadmodelfile = config['restartmodelfile']
  
  # if loadmodelfile or restartmodelfile specified, use its config
  if loadmodelfile is not None:
    load_config_from_model_file(loadmodelfile,config)
  elif restartmodelfile is not None:
    no_overwrite = ['num_train_epochs',]
    load_config_from_model_file(restartmodelfile,config,no_overwrite=no_overwrite)
  
  print(f"batch size = {config['batch_size']}")

  # seed the random number generators
  np.random.seed(config['numpy_seed'])
  torch.manual_seed(config['torch_seed'])
  
  # set device (cuda/cpu)
  device = torch.device(config['device'])

  plt.ion()

  data = None
  valdata = None
  if doloadtmpsavefile:
    # load cached, pre-chunked data  
    print(f'Loading tmp save file {tmpsavefile}')
    with open(tmpsavefile,'rb') as f:
      tmp = pickle.load(f)
    data = tmp['data']
    scale_perfly = tmp['scale_perfly']
    valdata = tmp['valdata']
    val_scale_perfly = tmp['val_scale_perfly']
    X = tmp['X']
    valX = tmp['valX']
  elif isdebug and os.path.exists(debugdatafile):
    with open(debugdatafile,'rb') as f:
      tmp = pickle.load(f)
      data = tmp['data']
      scale_perfly = tmp['scale_perfly']
      valdata = tmp['valdata']
      val_scale_perfly = tmp['val_scale_perfly']
  else:
    # load raw data
    data,scale_perfly = load_and_filter_data(config['intrainfile'],config)
    valdata,val_scale_perfly = load_and_filter_data(config['invalfile'],config)
  
  # if using discrete cosine transform, create dct matrix
  # this didn't seem to work well, so probably won't use in the future
  if config['dct_tau'] is not None and config['dct_tau'] > 0:
    dct_m,idct_m = get_dct_matrix(config['dct_tau'])
  else:
    dct_m = None
    idct_m = None

  # how much to pad outputs by -- depends on how many frames into the future we will predict
  npad = compute_npad(config['tspred_global'],dct_m)
  chunk_data_params = {'npad': npad}
  
  compute_feature_params = {
     "simplify_out": config['simplify_out'],
     "simplify_in": config['simplify_in'],
     "dct_m": dct_m,
     "tspred_global": config['tspred_global'],
     "compute_pose_vel": config['compute_pose_vel'],
     "discreteidx": config['discreteidx'],
  }

  # function for computing features
  reparamfun = lambda x,id,flynum,**kwargs: compute_features(x,id,flynum,scale_perfly,outtype=np.float32,
                                                            **compute_feature_params,**kwargs)

  val_reparamfun = lambda x,id,flynum,**kwargs: compute_features(x,id,flynum,val_scale_perfly,
                                                                outtype=np.float32,
                                                                **compute_feature_params,**kwargs)
  
  # sanity check on computing features when predicting many frames into the future
  sanity_check_tspred(data,compute_feature_params,npad,scale_perfly,contextl=config['contextl'],t0=510,flynum=0)
                                    
  if not doloadtmpsavefile:
    # chunk the data if we didn't load the pre-chunked cache file
    print('Chunking training data...')
    X = chunk_data(data,config['contextl'],reparamfun,**chunk_data_params)
    print('Chunking val data...')
    valX = chunk_data(valdata,config['contextl'],val_reparamfun,**chunk_data_params)
    print('Done.')
    
    if len(tmpsavefile) > 0:
      print('Saving chunked data to file')
      with open(tmpsavefile,'wb') as f:
        pickle.dump({'X': X,'valX': valX, 'data': data, 'valdata': valdata, 'scale_perfly': scale_perfly,'val_scale_perfly': val_scale_perfly},f)
      print('Done.')

  dataset_params = {
    'max_mask_length': config['max_mask_length'],
    'pmask': config['pmask'],
    'masktype': config['masktype'],
    'simplify_out': config['simplify_out'],
    'pdropout_past': config['pdropout_past'],
    'input_labels': config['input_labels'],
    'dozscore': True,
    'discreteidx': config['discreteidx'],
    'discretize_nbins': config['discretize_nbins'],
    'discretize_epsilon': config['discretize_epsilon'],
    'flatten_labels': config['flatten_labels'],
    'flatten_obs_idx': config['flatten_obs_idx'],
    'flatten_do_separate_inputs': config['flatten_do_separate_inputs'],
    'p_add_input_noise': config['p_add_input_noise'],
    'dct_ms': (dct_m,idct_m),
    'tspred_global': config['tspred_global'],
    'discrete_tspred': config['discrete_tspred'],
    'compute_pose_vel': config['compute_pose_vel'],
  }
  train_dataset_params = {
    'input_noise_sigma': config['input_noise_sigma'],
  }

  print('Creating training data set...')
  train_dataset = FlyMLMDataset(X,**train_dataset_params,**dataset_params)
  print('Done.')

  # zscore and discretize parameters for validation data set based on train data
  # we will continue to use these each time we rechunk the data
  dataset_params['zscore_params'] = train_dataset.get_zscore_params()
  dataset_params['discretize_params'] = train_dataset.get_discretize_params()

  print('Creating validation data set...')
  val_dataset = FlyMLMDataset(valX,**dataset_params)
  print('Done.')

  # get properties of examples from the first training example
  example = train_dataset[0]
  d_input = example['input'].shape[-1]
  d_output = train_dataset.d_output
  outnames = train_dataset.get_outnames()

  train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=config['batch_size'],
                                                shuffle=True,
                                                pin_memory=True,
                                                )
  ntrain = len(train_dataloader)

  val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=config['batch_size'],
                                              shuffle=False,
                                              pin_memory=True,
                                              )
  nval = len(val_dataloader)

  example = next(iter(train_dataloader))
  sz = example['input'].shape
  print(f'batch input shape = {sz}')

  # set up debug plots
  debug_params = {}
  # if contextl is long, still just look at samples from the first 64 frames
  if config['contextl'] > 64:
    debug_params['tsplot'] = np.round(np.linspace(0,64,5)).astype(int)
    debug_params['traj_nsamplesplot'] = 1
  hdebug = {}
  hdebug['train'] = initialize_debug_plots(train_dataset,train_dataloader,data,name='Train',**debug_params)
  hdebug['val'] = initialize_debug_plots(val_dataset,val_dataloader,valdata,name='Val',**debug_params)

  # create the model
  model,criterion = initialize_model(config,train_dataset,device)

  # optimizer
  num_training_steps = config['num_train_epochs'] * ntrain
  optimizer = transformers.optimization.AdamW(model.parameters(),**config['optimizer_args'])
  lr_scheduler = transformers.get_scheduler('linear',optimizer,num_warmup_steps=0,
                                            num_training_steps=num_training_steps)


  # initialize structure to keep track of loss
  loss_epoch = initialize_loss(train_dataset,config)
  last_val_loss = None

  progress_bar = tqdm.tqdm(range(num_training_steps))

  # create attention mask
  contextl = example['input'].shape[1]
  if config['modeltype'] == 'mlm':
    train_src_mask = generate_square_full_mask(contextl).to(device)
    is_causal = False
  elif config['modeltype'] == 'clm':
    train_src_mask = torch.nn.Transformer.generate_square_subsequent_mask(contextl,device=device)
    is_causal = True
    #train_src_mask = generate_square_subsequent_mask(contextl).to(device)
  else:
    raise

  # sanity check on temporal dependences
  sanity_check_temporal_dep(train_dataloader,device,train_src_mask,is_causal,model,tmess=300)

  modeltype_str = get_modeltype_str(config,train_dataset)
  if ('model_nickname' in config) and (config['model_nickname'] is not None):
    modeltype_str = config['model_nickname']

  hloss = initialize_loss_plots(loss_epoch)
  
  # epoch = 40
  # restartmodelfile = f'llmnets/flyclm_flattened_mixed_71G01_male_epoch{epoch}_20230517T153613.pth'
  # loss_epoch = load_model(restartmodelfile,model,device,lr_optimizer=optimizer,scheduler=lr_scheduler)
  # with torch.no_grad():
  #   pred = model(example['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
  # update_debug_plots(hdebug['train'],config,model,train_dataset,example,pred,name='Train',criterion=criterion)

  # train
  if loadmodelfile is None:
    
    # restart training
    if restartmodelfile is not None:
      loss_epoch = load_model(restartmodelfile,model,device,lr_optimizer=optimizer,scheduler=lr_scheduler)
      update_loss_nepochs(loss_epoch,config['num_train_epochs'])
      update_loss_plots(hloss,loss_epoch)
      #loss_epoch = {k: v.cpu() for k,v in loss_epoch.items()}
      epoch = np.nonzero(np.isnan(loss_epoch['train'].cpu().numpy()))[0][0]
      progress_bar.update(epoch*ntrain)
    else:
      epoch = 0
    
    savetime = datetime.datetime.now()
    savetime = savetime.strftime('%Y%m%dT%H%M%S')
    ntimepoints_per_batch = train_dataset.ntimepoints
    valexample = next(iter(val_dataloader))
    
    for epoch in range(epoch,config['num_train_epochs']):
      
      model.train()
      tr_loss = torch.tensor(0.0).to(device)
      if train_dataset.discretize:
        tr_loss_discrete = torch.tensor(0.0).to(device)
        tr_loss_continuous = torch.tensor(0.0).to(device)

      nmask_train = 0
      for step, example in enumerate(train_dataloader):
        
        pred = model(example['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
        loss,loss_discrete,loss_continuous = criterion_wrapper(example,pred,criterion,train_dataset,config)
          
        loss.backward()
        
        # how many timepoints are in this batch for normalization
        if config['modeltype'] == 'mlm':
          nmask_train += torch.count_nonzero(example['mask'])
        else:
          nmask_train += example['input'].shape[0]*ntimepoints_per_batch 

        if step % config['niterplot'] == 0:
          
          with torch.no_grad():
            trainpred = model.output(example['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
            valpred = model.output(valexample['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
          update_debug_plots(hdebug['train'],config,model,train_dataset,example,trainpred,name='Train',criterion=criterion,**debug_params)
          update_debug_plots(hdebug['val'],config,model,val_dataset,valexample,valpred,name='Val',criterion=criterion,**debug_params)
          plt.show()
          plt.pause(.1)

        tr_loss_step = loss.detach()
        tr_loss += tr_loss_step
        if train_dataset.discretize:
          tr_loss_discrete += loss_discrete.detach()
          tr_loss_continuous += loss_continuous.detach()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'])
        optimizer.step()
        lr_scheduler.step()
        model.zero_grad()
        
        # update progress bar
        stat = {'train loss': tr_loss.item()/nmask_train,'last val loss': last_val_loss,'epoch': epoch}
        if train_dataset.discretize:
          stat['train loss discrete'] = tr_loss_discrete.item()/nmask_train
          stat['train loss continuous'] = tr_loss_continuous.item()/nmask_train
        progress_bar.set_postfix(stat)
        progress_bar.update(1)
        
        # end of iteration loop

      # training epoch complete
      loss_epoch['train'][epoch] = tr_loss.item() / nmask_train
      if train_dataset.discretize:
        loss_epoch['train_discrete'][epoch] = tr_loss_discrete.item() / nmask_train
        loss_epoch['train_continuous'][epoch] = tr_loss_continuous.item() / nmask_train
      
      # compute validation loss after this epoch
      if val_dataset.discretize:
         loss_epoch['val'][epoch],loss_epoch['val_discrete'][epoch],loss_epoch['val_continuous'][epoch] = \
           compute_loss(model,val_dataloader,val_dataset,device,train_src_mask,criterion,config)
      else:
        loss_epoch['val'][epoch] = \
          compute_loss(model,val_dataloader,val_dataset,device,train_src_mask,criterion,config)
      last_val_loss = loss_epoch['val'][epoch].item()
    
      update_loss_plots(hloss,loss_epoch)
      plt.show()
      plt.pause(.1)

      # rechunk the training data
      if np.mod(epoch+1,config['epochs_rechunk']) == 0:
        print(f'Rechunking data after epoch {epoch}')
        X = chunk_data(data,config['contextl'],reparamfun,**chunk_data_params)
      
        train_dataset = FlyMLMDataset(X,**train_dataset_params,**dataset_params)
        print('New training data set created')

      if (epoch+1)%config['save_epoch'] == 0:
        savefile = os.path.join(config['savedir'],f"fly{modeltype_str}_epoch{epoch+1}_{savetime}.pth")
        print(f'Saving to file {savefile}')
        save_model(savefile,model,lr_optimizer=optimizer,scheduler=lr_scheduler,loss=loss_epoch,config=config)

    savefile = os.path.join(config['savedir'],f'fly{modeltype_str}_epoch{epoch+1}_{savetime}.pth')
    save_model(savefile,model,lr_optimizer=optimizer,scheduler=lr_scheduler,loss=loss_epoch,config=config)

    print('Done training')
  else:
    modeltype_str,savetime = parse_modelfile(loadmodelfile)
    loss_epoch = load_model(loadmodelfile,model,device,lr_optimizer=optimizer,scheduler=lr_scheduler)
    update_loss_plots(hloss,loss_epoch)
    
  model.eval()

  # compute predictions and labels for all validation data using default masking
  all_pred,all_labels = predict_all(val_dataloader,val_dataset,model,config,train_src_mask)

  # plot comparison between predictions and labels on validation data
  # predv = stack_batch_list(all_pred)
  # labelsv = stack_batch_list(all_labels)
  # maskv = stack_batch_list(all_mask)
  # pred_discretev = stack_batch_list(all_pred_discrete)
  # labels_discretev = stack_batch_list(all_labels_discrete)
  
  fig,ax = debug_plot_global_histograms(all_pred,all_labels,train_dataset,nbins=25,subsample=1,compare='pred')
  # glabelsv = train_dataset.get_global_movement(labelsv)
  # gpredprev = torch.zeros(glabelsv.shape)
  # gpredprev[:] = np.nan
  # for i,dt in enumerate(train_dataset.tspred_global):
  #   gpredprev[dt:,i,:] = glabelsv[:-dt,i,:]
  # predprev = torch.zeros(labelsv.shape)
  # predprev[:] = np.nan
  # train_dataset.set_global_movement(gpredprev,predprev)
  # fig,ax = debug_plot_global_histograms(predprev,labelsv,train_dataset,nbins=25,subsample=100)
  
  if train_dataset.dct_m is not None:
    debug_plot_dct_relative_error(all_pred,all_labels,train_dataset)
  if train_dataset.ntspred_global > 1:
    debug_plot_global_error(all_pred,all_labels,train_dataset)

  # crop to nplot for plotting
  nplot = min(len(all_labels),8000//config['batch_size']//config['contextl']+1)
  
  ntspred_plot = np.minimum(4,train_dataset.ntspred_global)
  featidxplot,ftplot = all_labels[0].select_featidx_plot(ntspred_plot)
  naxc = np.maximum(1,int(np.round(len(featidxplot)/nfeatures)))
  fig,ax = debug_plot_predictions_vs_labels(all_pred[:nplot],all_labels[:nplot],naxc=naxc,featidxplot=featidxplot)
  if train_dataset.ntspred_global > 1:
    featidxplot,ftplot = all_labels[0].select_featidx_plot(ntsplot=train_dataset.ntspred_global,ntsplot_relative=0)
    naxc = np.maximum(1,int(np.round(len(featidxplot)/nfeatures)))
    fig,ax = debug_plot_predictions_vs_labels(all_pred[:nplot],all_labels[:nplot],naxc=naxc,featidxplot=featidxplot)
    featidxplot,_ = all_labels[0].select_featidx_plot(ntsplot=1,ntsplot_relative=1)
    fig,ax = debug_plot_predictions_vs_labels(all_pred[:nplot],all_labels[:nplot],naxc=naxc,featidxplot=featidxplot)

  # if train_dataset.dct_tau > 0:
  #   fstrs = ['left_middle_leg_tip_angle','left_front_leg_tip_angle','left_wing_angle']
  #   fs = [mabe.posenames.index(x) for x in fstrs]
  #   featidxplot = train_dataset.ravel_label_index([(f,i+1) for i in range(train_dataset.dct_tau+1) for f in fs])
  #   fig,ax = debug_plot_predictions_vs_labels(predv,labelsv,pred_discretev,labels_discretev,outnames=outnames,maskidx=maskidx,featidxplot=featidxplot,dataset=val_dataset,naxc=len(fs))

  #   predrelative_dct = train_dataset.get_relative_movement_dct(predv.numpy())
  #   labelsrelative_dct = train_dataset.get_relative_movement_dct(labelsv.numpy())
  #   fsdct = [np.array(mabe.posenames)[featrelative].tolist().index(x) for x in fstrs]
  #   predrelative_dct = predrelative_dct[:,:,fsdct].astype(train_dataset.dtype)
  #   labelsrelative_dct = labelsrelative_dct[:,:,fsdct].astype(train_dataset.dtype)
  #   outnamescurr = [f'{f}_dt{i+1}' for i in range(train_dataset.dct_tau) for f in fstrs]
  #   fig,ax = debug_plot_predictions_vs_labels(torch.as_tensor(predrelative_dct.reshape((-1,train_dataset.dct_tau*len(fsdct)))),
  #                                             torch.as_tensor(labelsrelative_dct.reshape((-1,train_dataset.dct_tau*len(fsdct)))),
  #                                             outnames=outnamescurr,maskidx=maskidx,naxc=len(fstrs))


  # generate an animation of open loop prediction
  tpred = np.minimum(2000 + config['contextl'],valdata['isdata'].shape[0]//2)

  # all frames must have real data
  
  burnin = config['contextl']-1
  contextlpad = burnin + train_dataset.ntspred_max
  allisdata = interval_all(valdata['isdata'],contextlpad)
  isnotsplit = interval_all(valdata['isstart']==False,tpred)[1:,...]
  canstart = np.logical_and(allisdata[:isnotsplit.shape[0],:],isnotsplit)
  flynum = 3 # 2
  t0 = np.nonzero(canstart[:,flynum])[0]
  idxstart = np.minimum(40000,len(t0)-1)
  if len(t0) > idxstart:
    t0 = t0[idxstart]
  else:
    t0 = t0[0]
  #t0 = np.nonzero(canstart[:,flynum])[0][40000]    
  # flynum = 2
  # t0 = np.nonzero(canstart[:,flynum])[0][0]
  fliespred = np.array([flynum,])

  randstate_np = np.random.get_state()
  randstate_torch = torch.random.get_rng_state()

  nsamplesfuture = 32  
  
  # reseed numpy random number generator with randstate_np
  np.random.set_state(randstate_np)
  # reseed torch random number generator with randstate_torch
  torch.random.set_rng_state(randstate_torch)
  ani = animate_predict_open_loop(model,val_dataset,valdata,val_scale_perfly,config,fliespred,t0,tpred,debug=False,
                                  plotattnweights=False,plotfuture=train_dataset.ntspred_global>1,nsamplesfuture=nsamplesfuture)

  vidtime = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  savevidfile = os.path.join(config['savedir'],f"samplevideo_{modeltype_str}_{savetime}_{vidtime}.gif")

  print('Saving animation to file %s...'%savevidfile)
  writer = animation.PillowWriter(fps=30)
  ani.save(savevidfile,writer=writer)
  print('Finished writing.')

def debug_plot_histograms(dataset,alpha=1):
  r = np.random.rand(dataset.discretize_bin_samples.shape[0])-.5
  ftidx = dataset.unravel_label_index(dataset.discreteidx)
  #ftidx[featrelative[ftidx[:,0]],1]+=1
  fs = np.unique(ftidx[:,0])
  ts = np.unique(ftidx[:,1])
  nfs = len(fs)
  fig,ax = plt.subplots(nfs,1,sharey=True)
  fig.set_figheight(17)
  fig.set_figwidth(30)
  colors = mabe.get_n_colors_from_colormap('hsv',dataset.discretize_nbins)
  colors[:,:-1] *= .7
  colors = colors[np.random.permutation(dataset.discretize_nbins),:]
  colors[:,-1] = alpha
  edges = np.zeros((len(fs),2))
  edges[:,0] = np.inf
  edges[:,1] = -np.inf
  bin_edges = dataset.discretize_bin_edges
  bin_samples = dataset.discretize_bin_samples
  if dataset.sig_labels is not None:
    bin_edges = unzscore(bin_edges,dataset.mu_labels[dataset.discreteidx,None],dataset.sig_labels[dataset.discreteidx,None])
    bin_samples = unzscore(bin_samples,dataset.mu_labels[None,dataset.discreteidx,None],dataset.sig_labels[None,dataset.discreteidx,None])
  for i,idx in enumerate(dataset.discreteidx):
    f = ftidx[i,0]
    t = ftidx[i,1]
    fi = np.nonzero(fs==f)[0][0]
    ti = np.nonzero(ts==t)[0][0]
    edges[fi,0] = np.minimum(edges[fi,0],bin_edges[i,0])
    edges[fi,1] = np.maximum(edges[fi,1],bin_edges[i,-1])
    for j in range(dataset.discretize_nbins):
      ax[fi].plot(bin_samples[:,i,j],ti+r,'.',ms=.01,color=colors[j])
      ax[fi].plot([bin_edges[i,j],]*2,ti+np.array([-.5,.5]),'k-')
    ax[fi].plot([bin_edges[i,-1],]*2,ti+np.array([-.5,.5]),'k-')
    ax[fi].plot(bin_edges[i,[0,-1]],[ti+.5,]*2,'k-')
    ax[fi].plot(bin_edges[i,[0,-1]],[ti-.5,]*2,'k-')
  fnames = dataset.get_movement_names()
  for i,f in enumerate(fs):
    ax[i].set_title(fnames[f])
    ax[i].set_xlim(edges[i,0],edges[i,1])
    ax[i].set_yticks(np.arange(len(ts)))
    ax[i].set_yticklabels([str(t) for t in ts])
    ax[i].set_ylim(-.5,len(ts)-.5)
    ax[i].set_xscale('symlog')
  ax[-1].set_ylabel('Delta t')
  fig.tight_layout()
  return

def debug_plot_global_histograms(all_pred,all_labels,train_dataset,nbins=50,subsample=1,compare='time'):
  outnames_global = train_dataset.get_movement_names_global()
  
  # global labels, continuous representation, unzscored
  # ntimepoints x tspred x nglobal
  unz_glabelsv = np.concatenate([labels.get_future_global(zscored=False,use_todiscretize=True) for labels in all_labels],axis=0)

  # global predictions, continuous representation, unzscored
  # ntimepoints x tspred x nglobal
  unz_gpredv = np.concatenate([pred.get_future_global(zscored=False) for pred in all_pred],axis=0)

  if train_dataset.discretize:

    bin_edges = train_dataset.get_bin_edges(zscored=False)
    ftidx = all_labels[0].idx_multi_to_multifeattpred[all_labels[0].idx_multidiscrete_to_multi]
    bins = []
    for f in featglobal:
      j = np.nonzero(np.all(ftidx == np.array([f,1])[None,...],axis=1))[0][0]
      bins.append(bin_edges[j])
    nbins = train_dataset.discretize_nbins
  else:
    lims = [[np.percentile(unz_glabelsv[::100,:,axi].flatten(),i).item() for i in [.1,99.9]] for axi in range(nglobal)]
    bins = [np.arange(l[0],l[1],nbins+1) for l in lims]

  ntspred = len(train_dataset.tspred_global)
  off0 = .1
  
  if compare == 'time':
    colors = mabe.get_n_colors_from_colormap('jet',len(train_dataset.tspred_global))
    colors[:,:-1] *=.8

    fig,ax = plt.subplots(2,nglobal,figsize=(30,10),sharex='col')
    w = (1-2*off0)/ntspred
    for axj,(datacurr,datatype) in enumerate(zip([unz_glabelsv,unz_gpredv],['label ','pred '])):
      for axi in range(nglobal):
        ax[axj,axi].cla()
        off = off0
        for i in range(unz_glabelsv.shape[1]):
          density,_ = np.histogram(datacurr[::subsample,i,axi],bins=bins[axi],density=True)
          ax[axj,axi].bar(np.arange(nbins)+off,density,width=w,color=colors[i],log=True,
                          align='edge',label=str(train_dataset.tspred_global[i]))
          off+=w
        ax[axj,axi].set_xticks(np.arange(nbins+1))
        ax[axj,axi].set_xticklabels(['%.2f'%x for x in bins[axi]],rotation=90)
        ax[axj,axi].set_title(datatype+outnames_global[axi])
  elif compare == 'pred':
    colors = [[0,.5,.8],[.8,.5,.8]]

    fig,ax = plt.subplots(ntspred,nglobal,figsize=(20,30),sharex='col',sharey='all')
    w = (1-2*off0)/2
    for ti in range(ntspred):
      for fi in range(nglobal):
        axcurr = ax[ti,fi]
        axcurr.cla()
        off = off0
        for i,(datacurr,datatype) in enumerate(zip([unz_glabelsv,unz_gpredv],['label','pred'])):
          density,_ = np.histogram(datacurr[::subsample,ti,fi],bins=bins[fi],density=True)
          axcurr.bar(np.arange(nbins)+off,density,width=w,color=colors[i],log=False,
                     align='edge',label=datatype)
          off+=w
        axcurr.set_xticks(np.arange(nbins+1))
        axcurr.set_xticklabels(['%.2f'%x for x in bins[fi]],rotation=90)
        axcurr.set_title(f'{outnames_global[fi]} t = {train_dataset.tspred_global[ti]}')

  ax[0,0].legend()
  fig.tight_layout()
  
  return fig,ax

def debug_plot_global_error(all_pred,all_labels,train_dataset):
  """
  debug_plot_global_error(all_pred,all_labels,train_dataset)
  inputs:
  all_pred: list of PoseLabels objects containing predictions, each of shape (ntimepoints,d_output)
  all_labels: list of PoseLabels objects containing labels, each of shape (ntimepoints,d_output)
  train_dataset: FlyMLMDataset, the training dataset
  """
  outnames_global = train_dataset.get_movement_names_global()
  
    
  # global predictions, continuous representation, z-scored
  # nexamples x ntimepoints x tspred x nglobal
  gpredv = torch.tensor(np.stack([pred.get_future_global(zscored=True) for pred in all_pred],axis=0))
  
  # global labels, continuous representation
  # nexamples x ntimepoints x tspred x nglobal
  glabelsv = torch.tensor(np.stack([labels.get_future_global(zscored=True,use_todiscretize=True) for labels in all_labels],axis=0))
  
  nexamples = gpredv.shape[0]
  ntimepoints = gpredv.shape[1]
  ntspred = train_dataset.ntspred_global
  dglobal = all_labels[0].d_next_global
  
  # compute L1 error from continuous representations, all global features
  # network predictions
  errcont_all = torch.nn.L1Loss(reduction='none')(gpredv,glabelsv)
  errcont = np.nanmean(errcont_all,axis=(0,1))
  # just predicting zero (unzscored) all the time
  # only care about the global features
  pred0_obj = all_pred[0].copy_subindex(ts=np.array([0,]))
  pred0_obj.set_multi(np.zeros((1,pred0_obj.d_multi)),zscored=False)
  gpred0 = torch.tensor(pred0_obj.get_future_global(zscored=True)[None,...])
  err0cont_all = torch.nn.L1Loss(reduction='none')(gpred0,glabelsv)
  err0cont = np.nanmean(err0cont_all,axis=(0,1))
  
  # constant velocity predictions: use real labels from dt frames previous. 
  # note we we won't have predictions for the first dt frames
  gpredprev = torch.zeros(glabelsv.shape)
  gpredprev[:] = torch.nan
  for i,dt in enumerate(train_dataset.tspred_global):
    gpredprev[:,dt:,i,:] = glabelsv[:,:-dt,i,:]
  errprevcont_all = torch.nn.L1Loss(reduction='none')(gpredprev,glabelsv)
  errprevcont = np.nanmean(errprevcont_all,axis=(0,1))
  
  if train_dataset.discretize:
    # nexamples x ntimepoints x tspred x nglobal x nbins: discretized global predictions
    gpreddiscretev = torch.tensor(np.stack([pred.get_future_global_as_discrete() for pred in all_pred],axis=0))
    # nexamples x ntimepoints x tspred x nglobal x nbins: discretized global labels
    glabelsdiscretev = torch.tensor(np.stack([labels.get_future_global_as_discrete() for labels in all_labels],axis=0))
    # cross entropy error
    errdiscrete_all = torch.nn.CrossEntropyLoss(reduction='none')(gpreddiscretev.moveaxis(-1,1),
                                                                  glabelsdiscretev.moveaxis(-1,1))
    errdiscrete = np.nanmean(errdiscrete_all,axis=(0,1))

    gzerodiscretev = torch.tensor(np.tile(pred0_obj.get_future_global_as_discrete()[None,...],(nexamples,ntimepoints,1,1,1)))
    err0discrete_all = torch.nn.CrossEntropyLoss(reduction='none')(gzerodiscretev.moveaxis(-1,1),
                                                                   glabelsdiscretev.moveaxis(-1,1))
    err0discrete = np.nanmean(err0discrete_all,axis=(0,1))

    gpredprevdiscrete = torch.zeros(gpreddiscretev.shape,dtype=gpreddiscretev.dtype)
    gpredprevdiscrete[:] = torch.nan
    for i,dt in enumerate(train_dataset.tspred_global):
      gpredprevdiscrete[:,dt:,i,:,:] = glabelsdiscretev[:,:-dt,i,:,:]
    errprevdiscrete_all = torch.nn.CrossEntropyLoss(reduction='none')(gpredprevdiscrete.moveaxis(-1,1),
                                                                      glabelsdiscretev.moveaxis(-1,1))
    errprevdiscrete = np.nanmean(errprevdiscrete_all,axis=(0,1))

  if train_dataset.discretize:
    nc = 2
  else:
    nc = 1
  nr = nglobal
  fig,ax = plt.subplots(nr,nc,sharex=True,squeeze=False)
  fig.set_figheight(10)
  fig.set_figwidth(12)
  #colors = mabe.get_n_colors_from_colormap('viridis',train_dataset.dct_tau)
  for i in range(nglobal):
    ax[i,0].plot(errcont[:,i],'o-',label=f'Pred')
    ax[i,0].plot(err0cont[:,i],'s-',label=f'Zero')
    ax[i,0].plot(errprevcont[:,i],'s-',label=f'Prev')
    if train_dataset.discretize:
      ax[i,1].plot(errdiscrete[:,i],'o-',label=f'Pred')
      ax[i,1].plot(err0discrete[:,i],'s-',label=f'Zero')
      ax[i,1].plot(errprevdiscrete[:,i],'s-',label=f'Prev')
      ax[i,0].set_title(f'{outnames_global[i]} L1 error')
      ax[i,1].set_title(f'{outnames_global[i]} CE error')
    else:
      ax[i,0].set_title(outnames_global[i])
  ax[-1,-1].set_xticks(np.arange(train_dataset.ntspred_global))
  ax[-1,-1].set_xticklabels([str(t) for t in train_dataset.tspred_global])
  ax[-1,-1].set_xlabel('Delta t')
  ax[0,0].legend()
  plt.tight_layout()  
  
  return

def debug_plot_dct_relative_error(predv,labelsv,train_dataset):
  dt = train_dataset.dct_tau
  dcterr = np.sqrt(np.nanmean((predv[:,train_dataset.idxdct_relative]-labelsv[:,train_dataset.idxdct_relative])**2.,axis=0))
  dcterr0 = np.sqrt(np.nanmean((labelsv[:,train_dataset.idxdct_relative])**2.,axis=0))
  dcterrprev = np.sqrt(np.nanmean((labelsv[:-dt,train_dataset.idxdct_relative]-labelsv[dt:,train_dataset.idxdct_relative])**2.,axis=0))
  
  nc = int(np.ceil(np.sqrt(nrelative)))
  nr = int(np.ceil(nrelative/nc))
  fig,ax = plt.subplots(nr,nc,sharex=True,sharey=True)
  fig.set_figheight(14)
  fig.set_figwidth(23)
  ax = ax.flatten()
  for i in range(nrelative,nc*nr):
    ax[i].remove()
  ax = ax[:nrelative]
  for i in range(nrelative):
    ax[i].plot(dcterr[:,i],'o-',label=f'pred')
    ax[i].plot(dcterr0[:,i],'s-',label=f'zero')
    ax[i].plot(dcterrprev[:,i],'s-',label=f'prev')
    ax[i].set_title(mabe.posenames[np.nonzero(featrelative)[0][i]])
  ax[-1].set_xticks(np.arange(train_dataset.dct_tau))
  ax[(nc-1)*nr-1].set_xlabel('DCT feature')
  ax[0].legend()
  plt.tight_layout()  
  
  predrelative_dct = train_dataset.get_relative_movement_dct(predv.numpy())
  labelsrelative_dct = train_dataset.get_relative_movement_dct(labelsv.numpy())
  zpredrelative_dct = np.zeros(predrelative_dct.shape)
  zlabelsrelative_dct = np.zeros(labelsrelative_dct.shape)
  for i in range(predrelative_dct.shape[1]):
    zpredrelative_dct[:,i,:] = zscore(predrelative_dct[:,i,:],train_dataset.mu_labels[train_dataset.nextframeidx_relative],
                                      train_dataset.sig_labels[train_dataset.nextframeidx_relative])
    zlabelsrelative_dct[:,i,:] = zscore(labelsrelative_dct[:,i,:],train_dataset.mu_labels[train_dataset.nextframeidx_relative],
                                        train_dataset.sig_labels[train_dataset.nextframeidx_relative])
  idcterr = np.sqrt(np.nanmean((zpredrelative_dct-zlabelsrelative_dct)**2.,axis=0))
  nexterr = np.sqrt(np.nanmean((train_dataset.get_next_relative_movement(predv)-train_dataset.get_next_relative_movement(labelsv))**2,axis=0))
  err0 = np.sqrt(np.nanmean((zlabelsrelative_dct)**2,axis=0))
  errprev = np.sqrt(np.nanmean((zlabelsrelative_dct[:-dt,:,:]-zlabelsrelative_dct[dt:,:,:])**2,axis=0))
  
  plt.figure()
  plt.clf()
  plt.plot(idcterr[0,:],'s-',label='dct pred')
  plt.plot(nexterr,'o-',label='next pred')
  plt.plot(err0[0,:],'s-',label='zero')
  plt.plot(errprev[0,:],'s-',label='prev')
  plt.legend()
  plt.xticks(np.arange(nrelative))
  plt.gca().set_xticklabels([mabe.posenames[i] for i in np.nonzero(featrelative)[0]])
  plt.xticks(rotation=90)
  plt.title('Next frame prediction')
  plt.tight_layout()
  
  
  fig,ax = plt.subplots(nr,nc,sharex=True,sharey=True)
  fig.set_figheight(14)
  fig.set_figwidth(23)
  ax = ax.flatten()
  for i in range(nrelative,nc*nr):
    ax[i].remove()
  ax = ax[:nrelative]
  for i in range(nrelative):
    ax[i].plot(idcterr[:,i],'o-',label=f'pred')
    ax[i].plot(err0[:,i],'s-',label=f'zero')
    ax[i].plot(errprev[:,i],'s-',label=f'prev')
    ax[i].set_title(mabe.posenames[np.nonzero(featrelative)[0][i]])
  ax[-1].set_xticks(np.arange(train_dataset.dct_tau))
  ax[(nc-1)*nr-1].set_xlabel('Delta t')
  ax[0].legend()
  plt.tight_layout()  
  
  return

def debug_plot_histogram_edges(train_dataset):
  bin_edges = train_dataset.get_bin_edges(zscored=False)
  ftidx = train_dataset.unravel_label_index(train_dataset.discreteidx)
  fs = np.unique(ftidx[:,0])
  ts = np.unique(ftidx[:,1])
  fig,ax = plt.subplots(1,len(fs),sharey=True)
  movement_names = train_dataset.get_movement_names()
  for i,f in enumerate(fs):
    ax[i].cla()
    idx = ftidx[:,0]==f
    tscurr = ftidx[idx,1]
    tidx = mabe.npindex(ts,tscurr)
    ax[i].plot(bin_edges[idx,:],tidx,'.-')
    ax[i].set_title(movement_names[f])
    ax[i].set_xscale('symlog')
  ax[0].set_yticks(np.arange(len(ts)))
  ax[0].set_yticklabels([str(t) for t in ts])
  return fig,ax

if __name__ == "__main__":

  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-c',type=str,required=False,help='Path to config file',metavar='configfile',dest='configfile')
  parser.add_argument('-l',type=str,required=False,help='Path to model file to load',metavar='loadmodelfile',dest='loadmodelfile')
  parser.add_argument('-r',type=str,required=False,help='Path to model file to restart training from',metavar='restartmodelfile',dest='restartmodelfile')
  parser.add_argument('--clean',type=str,required=False,help='Delete intermediate networks saved in input directory.',metavar='cleandir',dest='cleandir')
  args = parser.parse_args()

  if args.cleandir is not None:
    assert os.path.isdir(args.cleandir)      
    removedfiles = clean_intermediate_results(args.cleandir)
  else:
    debug_fly_example(args.configfile,loadmodelfile=args.loadmodelfile,restartmodelfile=args.restartmodelfile)
    #main(args.configfile,loadmodelfile=args.loadmodelfile,restartmodelfile=args.restartmodelfile)
  #explore_representation(args.configfile)