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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm,animation,colors

import MABeFlyUtils as mabe

# local path to data -- modify this!
datadir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/seqdata20220307'
rootsavedir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/savednets'


def compute_straight_walk(X,thresh_bodyspeed=.125,thresh_angularspeed = 2.*np.pi/180.,
                          thresh_dist2wall=7.,thresh_boutlength = int(20./150.*mabe.FPS)):
  _,fthorax,thorax_theta = mabe.body_centric_kp(X)
  bodyspeed = np.sqrt(np.sum((fthorax[:,1:,:]-fthorax[:,:-1,:])**2.,axis=0))
  angularspeed = np.abs(mabe.modrange(thorax_theta[1:,:]-thorax_theta[:-1,:],-np.pi,np.pi))
  dist2wall = mabe.ARENA_RADIUS_MM-np.sqrt(np.min(np.sum(X[mabe.keypointidx,:,...]**2.,axis=1),axis=0))

  isstraightwalk = bodyspeed >= thresh_bodyspeed
  isstraightwalk = np.logical_and(isstraightwalk,angularspeed <= thresh_angularspeed)
  isstraightwalk = np.logical_and(isstraightwalk,dist2wall[:-1] >= thresh_dist2wall)
  
  isstraightwalk = scipy.ndimage.binary_opening(isstraightwalk,np.ones((thresh_boutlength-1,1)))
  
  nflies = isstraightwalk.shape[1]
  
  t0s = []
  t1s = []
  for i in range(nflies):
    pad = np.r_[False,isstraightwalk[:,i],False]
    t0scurr = np.flatnonzero(np.logical_and(pad[:-1]==False,pad[1:]==True))-2
    t1scurr = np.flatnonzero(np.logical_and(pad[:-1]==True,pad[1:]==False))-1
    t0s.append(t0scurr)
    t1s.append(t1scurr)
  
  return isstraightwalk,t0s,t1s,bodyspeed,angularspeed,dist2wall,fthorax,thorax_theta

def segment_strides(Xlegtip,t0s=None,t1s=None,filrad = 3,filwneg=-1.,filwpos=1.,peak_thresh=.05,
                    min_period=int(5./150.*mabe.FPS)):

  nflies = Xlegtip.shape[2]
  T = Xlegtip.shape[1]

  z = np.abs(filwneg) + np.abs(filwpos)
  filwneg = filwneg / z / 2. / filrad
  filwpos = filwpos / z / 2. / filrad
  # intentionally reversed because signal convolution is flipped!
  fil = np.vstack((np.zeros((filrad,1))+filwneg,np.zeros((filrad,1))+filwpos))

  if t0s is None:
    t0s = [0,]*nflies
  if t1s is None:
    t1s = [T-1,]*nflies

  # Xlegtip is 2 x T x nflies
  legtipspeed = np.sqrt(np.sum((Xlegtip[:,1:,...]-Xlegtip[:,:-1,...])**2.,axis=0))

  # 
  legtipacc = scipy.signal.convolve(legtipspeed,fil,mode='valid')

  peaks = []
  for fly in range(nflies):
    peaksfly = []
    for i in range(len(t0s[fly])):
      peakscurr,_ = scipy.signal.find_peaks(legtipacc[t0s[fly][i]:t1s[fly][i]+1,fly],height=peak_thresh,distance=min_period)
      if len(peakscurr) == 0:
        continue
      peakscurr += t0s[fly][i]
      peaksfly.append(peakscurr)
    peaks.append(np.array(peaksfly))

  return peaks


def compute_stride_error(Xlegtips,fthorax,thorax_theta,stride_ts):

  nflies = Xlegtips.shape[-1]
  nlegtips = Xlegtips.shape[0]

  nbouts = 0
  for fly in range(nflies):
    nbouts += np.count_nonzero(np.array(list(map(lambda x: x.size > 2,stride_ts[fly]))))

  mean_err_legtip = np.zeros(nflies)
  mean_err_legtip[:] = np.nan
  err_legtip = [None,]*nflies
  med_stride_period = None

  if nbouts == 0:
    return mean_err_legtip,err_legtip,med_stride_period

  stride_period = np.array([],dtype=int)
  for fly in range(nflies):
    for i in range(len(stride_ts[fly])):
      stride_period = np.r_[stride_period,(stride_ts[fly][i][1:]-stride_ts[fly][i][:-1])]
            
  med_stride_period = int(np.round(np.median(stride_period)))
            

  err_legtip = [np.zeros((nlegtips,med_stride_period,0)),]*nflies
  for fly in range(nflies):

    med_stride_period = int(np.round(np.median(stride_period)))


    nbouts = len(stride_ts[fly])
    for i in range(nbouts):
      ts = stride_ts[fly][i]
      nstrides = len(ts)
      if nstrides < 3:
        continue
      Xn = np.zeros((nlegtips,2,med_stride_period,nstrides-1))
      for j in range(nstrides-1):
        t0 = ts[j]
        t1 = t0 + med_stride_period
        Xn[...,j] = mabe.rotate_2d_points(Xlegtips[:,:,t0:t1,fly]-fthorax[None,:,[t0,],fly],thorax_theta[t0,fly])
      medXn = np.mean(Xn,axis=-1)
      errXn = np.sqrt(np.sum((Xn-medXn[...,None])**2,axis=1))
      err_legtip[fly] = np.concatenate((err_legtip[fly],errXn),axis=2)
    if err_legtip[fly].size > 0:
      mean_err_legtip[fly] = np.mean(err_legtip[fly])
      
  return mean_err_legtip,err_legtip,med_stride_period

def analyze_stride_deviation():

  # file containing the data
  xfile = os.path.join(datadir, 'Xusertrain_seq.npy')
  yfile = os.path.join(datadir, 'yusertrain_seq.npy')
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

  legtipidx = np.where(np.array([re.search(r'leg_tip',x) is not None for x in mabe.keypointnames]))[0]
  legtipkeypointnames = [mabe.keypointnames[x] for x in legtipidx]
  side1keypointnames = ['right_back_leg_tip', 'left_middle_leg_tip', 'right_front_leg_tip']
  side2keypointnames = ['left_back_leg_tip','right_middle_leg_tip','left_front_leg_tip']
  side1legtipidx = np.array([legtipkeypointnames.index(x) for x in  side1keypointnames])
  side2legtipidx = np.array([legtipkeypointnames.index(x) for x in  side2keypointnames])
  middle1idx = legtipkeypointnames.index('left_middle_leg_tip')
  middle2idx = legtipkeypointnames.index('right_middle_leg_tip')

  nlegtips = legtipidx.size

  all_mean_err_legtip = []
  all_err_legtip = []
  all_med_stride_period = []

  # select a sequence
  #seqi = 1

  for id in all_dataset.seqids:

    Xcurr = all_dataset.X[id]
    ycurr = all_dataset.y[id]

    isreal = mabe.get_real_flies(Xcurr.reshape((Xcurr.shape[0]*Xcurr.shape[1],)+Xcurr.shape[2:]))
    Xcurr = Xcurr[...,isreal]
    ycurr = ycurr[...,isreal]
    nflies = Xcurr.shape[-1]
    
    # find straight walks
    isstraightwalk,straight_t0s,straight_t1s,bodyspeed,angularspeed,dist2wall,fthorax,thorax_theta = compute_straight_walk(Xcurr)
    
    Xlegtip = Xcurr[legtipidx[middle1idx],...]
    
    # segment into strides
    stride_ts = segment_strides(Xlegtip,straight_t0s,straight_t1s)
    
    mean_err_legtip,err_legtip,med_stride_period = \
      compute_stride_error(Xcurr[legtipidx,...],
                          fthorax,thorax_theta,stride_ts)
      
    all_mean_err_legtip.append(mean_err_legtip)
    all_err_legtip.append(err_legtip)
    all_med_stride_period.append(med_stride_period)

    print(f'id: {id}, mean err: {np.nanmean(mean_err_legtip)}')
    
  # collect all the data
  x = np.array([])
  for i in range(len(all_err_legtip)):
    for fly in range(len(all_err_legtip[i])):
      if all_err_legtip[i][fly] is None or all_err_legtip[i][fly].size == 0:
        continue
      x = np.r_[x,all_err_legtip[i][fly].flatten()]
  
  nbins = 100
  maxx = np.percentile(x,99)
  medx = np.median(x)
  binedges = np.linspace(0.,maxx,nbins+1)
  frac,binedges = np.histogram(x,bins=binedges,density=False)
  frac = frac / sum(frac)
  bincenters = (binedges[1:]+binedges[:-1])/2.
  plt.figure()
  plt.plot(bincenters,frac)

  plt.xlabel('Deviation of leg tip (mm)')
  plt.ylabel('Fraction of frames')

  ylim = np.array(plt.gca().get_ylim())
  ylim[0] = 0.
  plt.plot([medx,medx],ylim)
  plt.gca().set_ylim(ylim)

  print(f'median error: {medx}')

  nstrides = np.zeros((len(all_err_legtip),maxnflies))
  nstrides[:] = np.nan
  for seqi in range(len(all_err_legtip)):
    for fly in range(len(all_err_legtip[seqi])):
      err_legtip = all_err_legtip[seqi][fly]
      if err_legtip is None or err_legtip.size == 0:
        nstrides[seqi,fly] = 0
      else:
        nstrides[seqi,fly] = err_legtip.shape[-1]
  
  maxstrides = np.nanmax(nstrides)
  seqi,fly = np.where(nstrides==maxstrides)
  seqi = seqi[0]
  fly = fly[0]
  
  id = all_dataset.seqids[seqi]
  Xcurr = all_dataset.X[id]
  ycurr = all_dataset.y[id]

  isreal = mabe.get_real_flies(Xcurr.reshape((Xcurr.shape[0]*Xcurr.shape[1],)+Xcurr.shape[2:]))
  Xcurr = Xcurr[...,isreal]
  ycurr = ycurr[...,isreal]
  nflies = Xcurr.shape[-1]
  
  # find straight walks
  isstraightwalk,straight_t0s,straight_t1s,bodyspeed,angularspeed,dist2wall,fthorax,thorax_theta = compute_straight_walk(Xcurr)
  
  Xlegtip = Xcurr[legtipidx[middle1idx],...]
  
  # segment into strides
  stride_ts = segment_strides(Xlegtip,straight_t0s,straight_t1s)
  boutidx = np.where(np.array(list(map(lambda x: x.size > 0,stride_ts[fly]))))[0]
  nax = boutidx.size
  
  legtipspeed = np.sqrt(np.sum((Xcurr[legtipidx,:,1:,...]-Xcurr[legtipidx,:,:-1,...])**2.,axis=1))
  
  ylim = np.array([0,1.2])
  fig,ax = plt.subplots(nax,1)

  notmiddle = np.ones(nlegtips,dtype=bool)
  notmiddle[middle1idx] = False
  notmiddle = np.where(notmiddle)[0]
  
  for ii in range(nax):
    i = boutidx[ii]
    t0 = straight_t0s[fly][i]
    t1 = straight_t1s[fly][i]
    h = ax[ii].plot(np.arange(t0,t1+1),legtipspeed[notmiddle,t0:t1+1,fly].T,alpha=.5)
    h += ax[ii].plot(np.arange(t0,t1+1),legtipspeed[middle1idx,t0:t1+1,fly].T,'k.-',lw=2)
    ts = stride_ts[fly][i]
    ax[ii].plot(np.tile(ts,(2,1)),np.tile(ylim,(len(ts),1)).T,'k-')
    ax[ii].set_ylim(ylim)
    if ii == 0:
      ax[ii].legend(h,[legtipkeypointnames[x] for x in np.r_[notmiddle,middle1idx]])

  nbouts = 0
  boutidx = []
  for fly in range(nflies):
    boutidx.append(np.where(np.array(list(map(lambda x: x.size > 2,stride_ts[fly]))))[0])
    nbouts += boutidx[-1].size

  naxc = int(np.ceil(np.sqrt(nbouts)))
  naxr = int(np.ceil(nbouts/naxc))

  fig,ax = plt.subplots(naxr,naxc,sharex=True,sharey=True)
  legtipcolors = matplotlib.cm.tab10(np.linspace(0,1,nlegtips))
  ax = ax.flatten()
  
  axi = 0 
  for fly in range(nflies):
    for ii in range(len(boutidx[fly])):
      i = boutidx[fly][ii]
      ts = stride_ts[fly][i]
      for j in range(len(ts)-1):
        t = ts[j]
        Xn = mabe.rotate_2d_points(Xcurr[:,:,ts[j]:ts[j+1],fly]-fthorax[None,:,[t,],fly],thorax_theta[t,fly])
        all_dataset.plot_fly(Xn[:,:,0],color='k',ax=ax[axi])
        for k in range(nlegtips):
          ax[axi].plot(Xn[legtipidx[k],0,:],Xn[legtipidx[k],1,:],'.-',color=legtipcolors[k,:])
      axi += 1
  
  plt.show()
  return

if __name__ == "__main__":
  analyze_stride_deviation()
