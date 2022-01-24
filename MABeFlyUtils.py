"""
Useful functions for interacting with the files X<split>_seq.npy.

FlyDataset is a pytorch Dataset for loading and grabbing frames.
plot_fly(), plot_flies(), and animate_pose_sequence()
can be used to visualize a fly, a frame, and a sequence of frames.
"""

import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm,animation,colors

# local path to data -- modify this!
datadir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/seqdata20220118'

# data frame rate
FPS = 150.
# size of the arena the flies are enclosed in
ARENA_RADIUS_MM = 26.689

# names of our keypoint features
keypointnames = [
  'wing_left_x',
  'wing_right_x',
  'antennae_x_mm',
  'right_eye_x_mm',
  'left_eye_x_mm',
  'left_shoulder_x_mm',
  'right_shoulder_x_mm',
  'end_notum_x_mm',
  'end_abdomen_x_mm',
  'middle_left_b_x_mm',
  'middle_left_e_x_mm',
  'middle_right_b_x_mm',
  'middle_right_e_x_mm',
  'tip_front_right_x_mm',
  'tip_middle_right_x_mm',
  'tip_back_right_x_mm',
  'tip_back_left_x_mm',
  'tip_middle_left_x_mm',
  'tip_front_left_x_mm',
]

# edges to connect subsets of the keypoints that maybe make sense
skeleton_edge_names = [
  ('end_notum_x_mm','end_abdomen_x_mm'),
  ('middle_left_e_x_mm','tip_middle_right_x_mm'),
  ('middle_right_b_x_mm','middle_right_e_x_mm'),
  ('middle_right_e_x_mm','tip_middle_left_x_mm'),
  ('end_notum_x_mm','middle_right_b_x_mm'),
  ('middle_left_b_x_mm','middle_left_e_x_mm'),
  ('end_notum_x_mm','middle_left_b_x_mm'),
  ('left_shoulder_x_mm','end_notum_x_mm'),
  ('antennae_x_mm','right_eye_x_mm'),
  ('antennae_x_mm','end_notum_x_mm'),
  ('left_shoulder_x_mm','tip_front_left_x_mm'),
  ('right_shoulder_x_mm','tip_front_right_x_mm'),
  ('end_notum_x_mm','tip_back_left_x_mm'),
  ('end_notum_x_mm','tip_back_right_x_mm'),
  ('antennae_x_mm','left_eye_x_mm'),
  ('right_shoulder_x_mm','end_notum_x_mm'),
  ('end_notum_x_mm','wing_left_x'),
  ('end_notum_x_mm','wing_right_x')
]

"""
pytorch Dataset for loading the data (__init__) and grabbing frames.
Utilities for visualizing one frame of data also provided.
"""
class FlyDataset(Dataset):
  
  """
  data = FlyDataset(xfile)
  Load in the data from file xfile and initialize member variables
  Optional annotations from yfile
  """
  def __init__(self,xfile,yfile=None,ntgtsout=None,normalize=True):
    data = np.load(xfile,allow_pickle=True).item()
    # X is a dictionary keyed by random 20-character strings
    self.X = data['keypoints']
    # X[seqid] is nkeypoints x d=2 x seql x maxnflies x
    for k,x in self.X.items():
      self.X[k] = x.transpose((2,3,0,1))
    
    # seqids is a list of the 20-character strings for accessing sequences
    self.seqids = list(self.X.keys())
    
    # featurenames is a list of pairs of strings with a string descriptor
    # of each feature. featurenames[i][j] corresponds to X[seqid][:,:,i,j]
    self.featurenames = data['vocabulary']
    
    # number of sequences
    self.nseqs = len(self.X)
    
    # get the sizes of inputs
    firstx = list(self.X.values())[0]
    # sequence length in frames
    self.seqlength = firstx.shape[2]
    # middle frame of the sequence, used for ordering flies by distance to target fly
    self.ctrf = int(np.ceil(self.seqlength/2))
    # maximum number of flies
    self.ntgts = firstx.shape[3]
    # number of flies to output
    if ntgtsout is None:
      self.ntgtsout = self.ntgts
    else:
      self.ntgtsout = ntgtsout
    # number of features
    self.nfeatures = firstx.shape[0]
    # number of coordinates -- 2
    self.d = firstx.shape[1]
    
    # for z-scoring
    self.mu = None
    self.sig = None
    if normalize:
      self.zscore()
    
    # y is optional, and not None only if annotations are provided in yfile
    # y is a dictionary keyed by the same 20-character strings as X, with
    # X[seqid] corresponding to y[seqid]
    # y[seqid] is ncategories x seql x maxnflies
    self.y = None
    self.categorynames = None
    self.ncategories = 0
    if yfile is not None:
      data = np.load(yfile,allow_pickle=True).item()
      self.y = data['annotations']
      firsty = list(self.y.values())[0]
      self.ncategories = firsty.shape[0]
    
      if 'vocabulary' in data:
        # categorynames is a list of string descriptors of each category
        self.categorynames = data['vocabulary']
      
      # check that sizes match
      assert( len(self.y) == self.nseqs )
      assert( set(self.y.keys()) == set(self.X.keys()) )
      assert( firsty.shape[1] == self.seqlength )
      assert( firsty.shape[2] == self.ntgts )

    # features to use for computing inter-fly distances
    self.fidxdist = np.where(np.array(list(map(lambda x: x[0] in ['antennae_x_mm','end_notum_x_mm','end_abdomen_x_mm'],self.featurenames))))[0]
    # not all flies have data for each sequence
    self.nfliesperseq = {}
    i = 0
    self.idx2seqnum = np.zeros(self.nseqs*self.ntgts,dtype=np.int32)
    self.idx2tgt = np.zeros(self.nseqs*self.ntgts,dtype=np.int32)
    for k,v in self.X.items():
      seqi = self.seqids.index(k)
      # x is nfeatures x d x seqlength x ntgts
      tgtidx = np.where(np.all(np.isnan(self.X[k]),axis=(0,1,2))==False)[0]
      ncurr = len(tgtidx)
      self.idx2seqnum[i:i+ncurr] = seqi
      self.idx2tgt[i:i+ncurr] = tgtidx
      i += ncurr
    self.idx2seqnum = self.idx2seqnum[:i]
    self.n = i
    self.idx2tgt = self.idx2tgt[:i]
    
    self.mu = np.nan
    self.sig = np.nan
    
    # which feature numbers correspond to keypoints and are on the defined skeleton?
    featurenamesx = list(map(lambda x: x[0],self.featurenames))
    self.keypointidx = np.where(np.array(list(map(lambda x: x[0] in keypointnames,self.featurenames))))[0]
    self.skeleton_edges = np.zeros((len(skeleton_edge_names),2),dtype=int)
    for i in range(len(skeleton_edge_names)):
      for j in range(2):
        self.skeleton_edges[i,j] = featurenamesx.index(skeleton_edge_names[i][j])  
  
  def reformat_x(self,x):
    x = x.reshape((self.nfeatures*self.d,x.shape[2],x.shape[3]))
    return x
  
  def zscore(self):
    
    # compute mean
    # could compute std at same time, but want to avoid overflow
    self.mu = 0.
    n = 0.
    for x in self.X.values():
      isgood = np.isnan(x) == False
      self.mu += np.nansum(x,axis=(2,3))
      n += np.sum(isgood,axis=(2,3))
    self.mu /= n
    self.mu = self.mu.reshape((self.mu.shape[0],self.mu.shape[1],1,1))
    
    # compute standard deviation
    self.sig = 0.
    for x in self.X.values():
      self.sig += np.nansum( (x-self.mu)**2.,axis=(2,3) ) / n
    self.sig = np.sqrt(self.sig)
    self.sig = self.sig.reshape((self.mu.shape[0],self.mu.shape[1],1,1))
    
    for seqid in self.X:
      # normalize
      self.X[seqid] = (self.X[seqid]-self.mu)/self.sig
      # set nans to the mean value
      isreal = get_real_flies(self.X[seqid])
      replace = np.isnan(self.X[seqid])
      replace[:,:,:,isreal==False] = False
      self.X[seqid][replace] = 0.
      
    return

  # total number of sequence-fly pairs
  def __len__(self):
    return self.n
  
  """
  Unlabeled dataset:
  x,seqid,seqnum,tgt = data.getitem(idx)
  Labeled dataset:
  x,y,seqid,seqnum,tgt = data.getitem(idx)
  The getitem() function inputs an index between 0 and len(data)-1
  and returns a data example corresponding to a sequence (seqid or seqnum)
  and fly (tgt) corresponding to input idx.
  The output data x is reordered so that the selected fly is first, and
  other flies are ordered by distance to that fly.
  Input:
  idx: Integer between 0 and len(self)-1 specifying which sample to select.
  Output:
  x is an ndarray of size nkeypoints x 2 x seql x maxnflies input data sample
  y (only output for labeled dataset) is an ndarray of size
  ncategories x seqlength output data sample for target 0
  seqid: 20-character string identifying the selected sequence
  seqnum: integer identifying the selected sequence
  tgt: Integer identifying the selected fly.
  """
  def getitem(self,idx):
    seqnum = self.idx2seqnum[idx]
    tgt = self.idx2tgt[idx]
    seqid = self.seqids[seqnum]

    # sort flies by distance to target
    #nkpts = len(self.fidxdist)
    x = self.X[seqid]
    d = self.get_fly_dists(x, tgt)
    order = np.argsort(d)
    # x is nkeypoints x d x seqlength x ntgts
    x = x[...,order[:self.ntgtsout]]
    
    if self.y is None:
      return x,seqid,seqnum,tgt
    
    y = self.y[seqid][:,:,tgt]
    return x,y,seqid,seqnum,tgt
  
  """
  Unlabeled dataset:
  x = data[idx]
  Labeled dataset:
  x,y = data[idx]
  The __getitem__() function inputs an index between 0 and len(data)-1
  and returns a data example corresponding to a sequence (seqid or seqnum)
  and fly (tgt) corresponding to input idx.
  The output data x (and optionally y) are reordered so that the selected fly
  is first, and other flies are ordered by distance to that fly.
  Input:
  idx: Integer between 0 and len(self)-1 specifying which sample to select.
  Output:
  Dictionary with the following entries:
  x is an ndarray of size (nkeypoints*d) x seqlength x ntgts data sample
  y (only output for labeled dataset) is an ndarray of size
  ncategories x seqlength output data sample for target 0
  seqnum: integer identifying the selected sequence
  tgt: Integer identifying the selected fly.
  """
  def __getitem__(self,idx):
    if self.y is None:
      x,seqid,seqnum,tgt = self.getitem(idx)
      x = self.reformat_x(x)
      return {'x': x, 'seqnum': seqnum, 'tgt': tgt}
    x,y,seqid,seqnum,tgt = self.getitem(idx)
    x = self.reformat_x(x)
    return {'x': x, 'y': y, 'seqnum': seqnum, 'tgt': tgt}

  """
  d = data.get_fly_dists(x,tgt=0)
  Compute the distance between fly tgt and all other flies. This is defined as the
  minimum distance between any pair of the following keypoints:
  'antennae','end_notum_x','end_abdomen'
  at middle frame data.ctrf
  Input:
  x: ndarray of size nkeypoints x d x seqlength x ntgts data sample, sequence of data for all flies
  tgt: (optional) which fly to compute distances to. Default is 0.
  Output:
  d: Array of length nflies with the squared distance to the selected target.
  """
  def get_fly_dists(self, x, tgt=0):
    nkpts = len(self.fidxdist)
    d = np.min(np.sum((x[self.fidxdist,:,self.ctrf,:].reshape((nkpts,1,self.d,self.ntgts))-
                x[self.fidxdist,:,self.ctrf,tgt].reshape((1,nkpts,self.d,1)))**2.,axis=2),
               axis=(0,1))
    return d
  
  """
  hkpt,hedge,fig,ax = data.plot_fly(pose=None, idx=None, f=None,
                                    fig=None, ax=None, kptcolors=None, color=None, name=None,
                                    plotskel=True, plotkpts=True, hedge=None, hkpt=None)
  Visualize the single fly position specified by pose
  Inputs:
  pose: Optional. nfeatures x 2 ndarray. Default is None.
  idx: Data sample index to plot. Only used if pose is not input. Default: None.
  f: Frame of data sample idx to plot. Only used if pose is not input. Default: self.ctrf.
  fig: Optional. Handle to figure to plot in. Only used if ax is not specified. Default = None.
  ax: Optional. Handle to axes to plot in. Default = None.
  kptcolors: Optional. Color scheme for each keypoint. Can be a string defining a matplotlib
  colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to 'hsv'.
  Default: None
  color: Optional. Color for edges plotted. If None, it is set to [.6,.6,.6]. efault: None.
  name: Optional. String defining an identifying label for these plots. Default None.
  plotskel: Optional. Whether to plot skeleton edges. Default: True.
  plotkpts: Optional. Whether to plot key points. Default: True.
  hedge: Optional. Handle of edges to update instead of plot new edges. Default: None.
  hkpt: Optional. Handle of keypoints to update instead of plot new key points. Default: None.
  """
  def plot_fly(self, x=None, idx=None, f=None, **kwargs):
    if x is not None:
      return plot_fly(pose=x, kptidx=self.keypointidx, skelidx=self.skeleton_edges, **kwargs)
    else:
      assert(idx is not None)
      if f is None:
        f = self.ctrf
      x,seqid,seqnum,tgt = self[idx]
      return plot_fly(pose=x[f, 0, :, :], kptidx=self.keypointidx, skelidx=self.skeleton_edges, **kwargs)
    
  """
  hkpt,hedge,fig,ax = data.plot_flies(poses=None,, idx=None, f=None,
                                      colors=None,kptcolors=None,hedges=None,hkpts=None,
                                      **kwargs)
  Visualize all flies for a single frame specified by poses.
  Inputs:
  poses: Required. nfeatures x 2 x nflies ndarray.
  idx: Data sample index to plot. Only used if pose is not input. Default: None.
  f: Frame of data sample idx to plot. Only used if pose is not input. Default: self.ctrf.
  colors: Optional. Color scheme for edges plotted for each fly. Can be a string defining a matplotlib
  colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to the Dark3
  colormap I defined in get_Dark3_cmap(). Default: None.
  kptcolors: Optional. Color scheme for each keypoint. Can be a string defining a matplotlib
  colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to [0,0,0].
  Default: None
  hedges: Optional. List of handles of edges, one per fly, to update instead of plot new edges. Default: None.
  hkpts: Optional. List of handles of keypoints, one per fly,  to update instead of plot new key points.
  Default: None.
  Extra arguments: All other arguments will be passed directly to plot_fly.
  """
  def plot_flies(self, x=None, idx=None, f=None, **kwargs):
    if x is not None:
      return plot_flies(poses=x, kptidx=self.keypointidx, skelidx=self.skeleton_edges, **kwargs)
    else:
      assert(idx is not None)
      if f is None:
        f = self.ctrf
      x,_,_,_ = self.get_item(idx=idx)
      return plot_flies(x[:, :, f, :], self.keypointidx, self.skeleton_edges, **kwargs)
    
"""
dark3cm = get_Dark3_cmap()
Returns a new matplotlib colormap based on the Dark2 colormap.
I didn't have quite enough unique colors in Dark2, so I made Dark3 which
is Dark2 followed by all Dark2 colors with the same hue and saturation but
half the brightness.
"""
def get_Dark3_cmap():
  dark2 = list(cm.get_cmap('Dark2').colors)
  dark3 = dark2.copy()
  for c in dark2:
    chsv = colors.rgb_to_hsv(c)
    chsv[2] = chsv[2]/2.
    crgb = colors.hsv_to_rgb(chsv)
    dark3.append(crgb)
  dark3cm = colors.ListedColormap(tuple(dark3))
  return dark3cm

"""
isreal = get_real_flies(x)
Returns which flies in the input ndarray x correspond to real data (are not nan).
Input:
x: ndarray of arbitrary dimensions, as long as the last dimension corresponds to targets.
"""
def get_real_flies(x):
  # x is ... x ntgts
  dims = tuple(range(x.ndim-1))
  isreal = np.all(np.isnan(x),axis=dims)==False
  return isreal

"""
fig,ax,isnewaxis = set_fig_ax(fig=None,ax=None)
Create new figure and/or axes if those are not input.
Returns the handles to those figures and axes.
isnewaxis is whether a new set of axes was created.
"""
def set_fig_ax(fig=None,ax=None):
    if ax is None:
      if fig is None:
        fig = plt.figure(figsize=(8, 8))
      ax = fig.add_subplot(111)
      isnewaxis = True
    else:
      isnewaxis = False
    return fig, ax, isnewaxis

"""
hkpt,hedge,fig,ax = plot_fly(pose=None, kptidx=None, skelidx=None,
                             fig=None, ax=None, kptcolors=None, color=None, name=None,
                             plotskel=True, plotkpts=True, hedge=None, hkpt=None)
Visualize the single fly position specified by pose
Inputs:
pose: Required. nfeatures x 2 ndarray.
kptidx: Required. 1-dimensional array specifying which keypoints to plot
skelidx: Required. nedges x 2 ndarray specifying which keypoints to connect with edges
fig: Optional. Handle to figure to plot in. Only used if ax is not specified. Default = None.
If None, a new figure is created.
ax: Optional. Handle to axes to plot in. Default = None. If None, new axes are created.
kptcolors: Optional. Color scheme for each keypoint. Can be a string defining a matplotlib
colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to 'hsv'.
Default: None
color: Optional. Color for edges plotted. If None, it is set to [.6,.6,.6]. efault: None.
name: Optional. String defining an identifying label for these plots. Default None.
plotskel: Optional. Whether to plot skeleton edges. Default: True.
plotkpts: Optional. Whether to plot key points. Default: True.
hedge: Optional. Handle of edges to update instead of plot new edges. Default: None.
hkpt: Optional. Handle of keypoints to update instead of plot new key points. Default: None.
"""
def plot_fly(pose=None, kptidx=None, skelidx=None, fig=None, ax=None, kptcolors=None, color=None, name=None,
             plotskel=True, plotkpts=True, hedge=None, hkpt=None):
  # plot_fly(x,fig=None,ax=None,kptcolors=None):
  # x is nfeatures x 2
  assert(pose is not None)
  assert(kptidx is not None)
  assert(skelidx is not None)

  fig,ax,isnewaxis = set_fig_ax(fig=fig,ax=ax)
  isreal = get_real_flies(pose)
  
  if plotkpts:
    if isreal:
      xc = pose[kptidx,0]
      yc = pose[kptidx,1]
    else:
      xc = []
      yc = []
    if hkpt is None:
      if kptcolors is None:
        kptcolors = 'hsv'
      if (type(kptcolors) == list or type(kptcolors) == np.ndarray) and len(kptcolors) == 3:
        kptname = 'keypoints'
        if name is not None:
          kptname = name + ' ' + kptname
        hkpt = ax.plot(xc,yc,'.',color=kptcolors,label=kptname,zorder=10)[0]
      else:
        if type(kptcolors) == str:
          kptcolors = plt.get_cmap(kptcolors)
        hkpt = ax.scatter(xc,yc,c=np.arange(len(kptidx)),marker='.',cmap=kptcolors,zorder=10)
    else:
      if type(hkpt) == matplotlib.lines.Line2D:
        hkpt.set_data(xc,yc)
      else:
        hkpt.set_offsets(np.column_stack((xc,yc)))
  
  if plotskel:
    nedges = skelidx.shape[0]
    if isreal:
      xc = np.concatenate((pose[skelidx,0],np.zeros((nedges,1))+np.nan),axis=1)
      yc = np.concatenate((pose[skelidx,1],np.zeros((nedges,1))+np.nan),axis=1)
    else:
      xc = np.array([])
      yc = np.array([])
    if hedge is None:
      edgename = 'skeleton'
      if name is not None:
        edgename = name + ' ' + edgename
      if color is None:
        color = [.6,.6,.6]
      hedge = ax.plot(xc.flatten(),yc.flatten(),'-',color=color,label=edgename,zorder=0)[0]
    else:
      hedge.set_data(xc.flatten(),yc.flatten())

  if isnewaxis:
    ax.axis('equal')

  return hkpt,hedge,fig,ax
 
"""
hkpt,hedge,fig,ax = plot_flies(poses=None, kptidx=None, skelidx=None,
                               colors=None,kptcolors=None,hedges=None,hkpts=None,
                               **kwargs)
Visualize all flies for a single frame specified by poses.
Inputs:
poses: Required. nfeatures x 2 x nflies ndarray.
kptidx: Required. 1-dimensional array specifying which keypoints to plot
skelidx: Required. nedges x 2 ndarray specifying which keypoints to connect with edges
colors: Optional. Color scheme for edges plotted for each fly. Can be a string defining a matplotlib
colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to the Dark3
colormap I defined in get_Dark3_cmap(). Default: None.
kptcolors: Optional. Color scheme for each keypoint. Can be a string defining a matplotlib
colormap (e.g. 'hsv'), a matplotlib colormap, or a single color. If None, it is set to [0,0,0].
Default: None
hedges: Optional. List of handles of edges, one per fly, to update instead of plot new edges. Default: None.
hkpts: Optional. List of handles of keypoints, one per fly,  to update instead of plot new key points.
Default: None.
Extra arguments: All other arguments will be passed directly to plot_fly.
"""
def plot_flies(poses=None,fig=None,ax=None,colors=None,kptcolors=None,hedges=None,hkpts=None,**kwargs):

  fig,ax,isnewaxis = set_fig_ax(fig=fig,ax=ax)
  if colors is None:
    colors = get_Dark3_cmap()
  if kptcolors is None:
    kptcolors = [0,0,0]
  nflies = poses.shape[-1]
  if not (type(colors) == list or type(kptcolors) == np.ndarray):
    if type(colors) == str:
      cmap = cm.get_cmap(colors)
    else:
      cmap = colors
    colors = cmap(np.linspace(0.,1.,nflies))
    
  if hedges is None:
    hedges = [None,]*nflies
  if hkpts is None:
    hkpts = [None,]*nflies
    
  for fly in range(nflies):
    hkpts[fly],hedges[fly],fig,ax = plot_fly(poses[...,fly],fig=fig,ax=ax,color=colors[fly,...],
                                             kptcolors=kptcolors,hedge=hedges[fly],hkpt=hkpts[fly],**kwargs)
  if isnewaxis:
    ax.axis('equal')
  
  return hkpts,hedges,fig,ax

def plot_annotations(y=None,values=None,names=None,fig=None,ax=None,
                     patchcolors=None,binarycolors=False,color0=None,axcolor=[.7,.7,.7]):
  
  fig,ax,isnewaxis = set_fig_ax(fig=fig,ax=ax)
  ncategories = y.shape[0]
  seql = y.shape[1]
  if values is None:
    values = np.unique(y[np.isnan(y)==False])
  else:
    values = np.asarray(values)
  if values.dtype == bool:
    values = values.astype(int)
  if y.dtype == bool:
    y = y.astype(int)
  novalue = values[0]-1

  if binarycolors:
    if color0 is None:
      color0 = [1,1,1]
    if patchcolors is None:
      patchcolors = 'tab10'
    if type(patchcolors) == str:
      patchcolors = cm.get_cmap(patchcolors)
    if not (type(patchcolors) == list or type(patchcolors) == np.ndarray):
      patchcolors = patchcolors(np.linspace(0.,1.,ncategories))
    colors1 = patchcolors.copy()
  else:
  
    if patchcolors is None:
      patchcolors = 'tab10'
    if type(patchcolors) == str:
      patchcolors = cm.get_cmap(patchcolors)
    if not (type(patchcolors) == list or type(patchcolors) == np.ndarray):
      patchcolors = patchcolors(np.linspace(0.,1.,len(values)))
  
  hpatch = []
  for c in range(ncategories):
    ycurr = y[c,:]
    ycurr = np.insert(ycurr,0,novalue)
    ycurr = np.append(ycurr,novalue)
    hpatchcurr = []
    if binarycolors:
      patchcolors = np.vstack((color0,colors1[c,:3]))
    for i in range(len(values)):
      patchcolor = patchcolors[i,:]
      v = values[i]
      t0s = np.where(np.logical_and(ycurr[:-1] != v,ycurr[1:] == v))[0]-.5
      t1s = np.where(np.logical_and(ycurr[:-1] == v,ycurr[1:] != v))[0]-.5
      # xy should be nrects x 4 x 2
      xy = np.zeros((len(t0s),4,2))
      xy[:,0,0] = t0s
      xy[:,1,0] = t0s
      xy[:,2,0] = t1s
      xy[:,3,0] = t1s
      xy[:,0,1] = c-.5
      xy[:,1,1] = c+.5
      xy[:,2,1] = c+.5
      xy[:,3,1] = c-.5
      catpath = matplotlib.path.Path.make_compound_path_from_polys(xy)
      patch = matplotlib.patches.PathPatch(catpath,facecolor=patchcolor)
      hpatchcurr.append(ax.add_patch(patch))
    hpatch.append(hpatchcurr)
  
  ax.set_xlim([-1,seql])
  ax.set_ylim([-1,ncategories])
  ax.set_facecolor(axcolor)
  if names is not None and len(names) == ncategories:
    ax.set_yticks(np.arange(0,ncategories))
    ax.set_yticklabels(names)
  return hpatch,fig,ax

"""
animate_pose_sequence(seq=None, kptidx=None, skelidx=None,
                      start_frame=0,stop_frame=None,skip=1,
                      fig=None,ax=None,savefile=None,
                      **kwargs)
Visualize all flies for the input sequence of frames seq.
Inputs:
seq: Required. nfeatures x 2 x seql x nflies ndarray.
kptidx: Required. 1-dimensional array specifying which keypoints to plot
skelidx: Required. nedges x 2 ndarray specifying which keypoints to connect with edges
start_frame: Which frame of the sequence to start plotting at. Default: 0.
stop_frame: Which frame of the sequence to end plotting on. Default: None. If None, the
sequence length (seq.shape[0]) is used.
skip: How many frames to skip between plotting. Default: 1.
fig: Optional. Handle to figure to plot in. Only used if ax is not specified. Default = None.
If None, a new figure is created.
ax: Optional. Handle to axes to plot in. Default = None. If None, new axes are created.
savefile: Optional. Name of video file to save animation to. If None, animation is displayed
instead of saved.
Extra arguments: All other arguments will be passed directly to plot_flies.
"""
def animate_pose_sequence(seq=None,start_frame=0,stop_frame=None,skip=1,
                          fig=None,ax=None,
                          annotation_sequence=None,
                          savefile=None,
                          **kwargs):
    
  if stop_frame is None:
    stop_frame = seq.shape[2]
  fig,ax,isnewaxis = set_fig_ax(fig=fig,ax=ax)
  
  isreal = get_real_flies(seq)
  idxreal = np.where(np.any(isreal,axis=0))[0]
  seq = seq[...,idxreal]

  # plot the arena wall
  theta = np.linspace(0,2*np.pi,360)
  ax.plot(ARENA_RADIUS_MM*np.cos(theta),ARENA_RADIUS_MM*np.sin(theta),'k-',zorder=-10)
  minv = -ARENA_RADIUS_MM*1.01
  maxv = ARENA_RADIUS_MM*1.01
  
  # first frame
  f = start_frame
  h = {}
  h['kpts'],h['edges'],fig,ax = plot_flies(poses=seq[...,f,:],fig=fig,ax=ax,**kwargs)
  h['frame'] = plt.text(-ARENA_RADIUS_MM*.99,ARENA_RADIUS_MM*.99,'Frame %d (%.2f s)'%(f,float(f)/FPS),
                        horizontalalignment='left',verticalalignment='top')
  ax.set_xlim(minv,maxv)
  ax.set_ylim(minv,maxv)
  ax.axis('equal')
  ax.axis('off')
  fig.tight_layout(pad=0)
  #ax.margins(0)
  
  def update(f):
    nonlocal fig
    nonlocal ax
    nonlocal h
    h['kpts'],h['edges'],fig,ax = plot_flies(poses=seq[...,f,:],fig=fig,ax=ax,
                                             hedges=h['edges'],hkpts=h['kpts'],**kwargs)
    h['frame'].set_text('Frame %d (%.2f s)'%(f,float(f)/FPS))
    return h['edges']+h['kpts']
  
  ani = animation.FuncAnimation(fig, update, frames=np.arange(start_frame,stop_frame,skip,dtype=int))
  if savefile is not None:
    print('Saving animation to file %s...'%savefile)
    writer = animation.PillowWriter(fps=30)
    ani.save(savefile,writer=writer)
    print('Finished writing.')
  else:
    plt.show()
   
"""
DatasetTest()
Driver function to test loading and accessing data.
"""
def DatasetTest():
  
  # file containing the data
  xfile = os.path.join(datadir,'Xusertrain_seq.npy')
  assert(os.path.exists(xfile))
  
  # load data and initialize Dataset object
  datatrain = FlyDataset(xfile)
  
  # get one sample sequence.
  x,seqid,seqnum,tgt = datatrain.getitem(123)
  # which sequence and fly did this correspond to?
  print('x.shape = {shape}, seqid = {seqid}, seqnum = {seqnum}, tgt = {tgt}'.format(shape=str(x.shape),seqid=seqid,seqnum=seqnum,tgt=tgt))
  # make sure the flies are sorted by distance to the first fly.
  print('These distances should be decreasing, with nans (if any) at the end): %s' % str(datatrain.get_fly_dists(x)))
  #hkpts,hedges,fig,ax = datatrain.plot_flies(x[:,:,datatrain.ctrf,:],colors='hsv')
  
  # Set savefile if we want to save a video
  #savefile = 'seq.gif'
  savefile=None

  # plot that sequence
  animate_pose_sequence(seq=x,kptidx=datatrain.keypointidx,skelidx=datatrain.skeleton_edges,savefile=savefile)

if __name__ == "__main__":
  DatasetTest()
  