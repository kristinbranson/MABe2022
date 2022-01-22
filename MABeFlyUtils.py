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
  """
  def __init__(self,xfile):
    data = np.load(xfile,allow_pickle=True).item()
    self.X = data['keypoints']
    self.seqids = list(self.X.keys())
    self.featurenames = data['vocabulary']
    self.nseqs = len(self.X)
    firstx = list(self.X.values())[0]
    self.seqlength = firstx.shape[0]
    self.ctrf = int(np.ceil(self.seqlength/2))
    self.ntgts = firstx.shape[1]
    self.nfeatures = firstx.shape[2]
    self.d = firstx.shape[3] # this should be 2
    # features to use for computing inter-fly distances
    self.fidxdist = np.where(np.array(list(map(lambda x: x[0] in ['antennae_x_mm','end_notum_x_mm','end_abdomen_x_mm'],self.featurenames))))[0]
    # not all flies have data for each sequence
    self.nfliesperseq = {}
    i = 0
    self.idx2seqnum = np.zeros(self.nseqs*self.ntgts,dtype=np.uint32)
    self.idx2tgt = np.zeros(self.nseqs*self.ntgts,dtype=np.uint32)
    for k,v in self.X.items():
      seqi = self.seqids.index(k)
      tgtidx = np.where(np.all(np.isnan(self.X[k]),axis=(0,2,3))==False)[0]
      ncurr = len(tgtidx)
      self.idx2seqnum[i:i+ncurr] = seqi
      self.idx2tgt[i:i+ncurr] = tgtidx
      i += ncurr
    self.idx2seqnum = self.idx2seqnum[:i]
    self.n = i
    self.idx2tgt = self.idx2tgt[:i]
    
    # which feature numbers correspond to keypoints and are on the defined skeleton?
    featurenamesx = list(map(lambda x: x[0],self.featurenames))
    self.keypointidx = np.where(np.array(list(map(lambda x: x[0] in keypointnames,self.featurenames))))[0]
    self.skeleton_edges = np.zeros((len(skeleton_edge_names),2),dtype=int)
    for i in range(len(skeleton_edge_names)):
      for j in range(2):
        self.skeleton_edges[i,j] = featurenamesx.index(skeleton_edge_names[i][j])
  
  # total number of sequence-fly pairs
  def __len__(self):
    return self.n
  
  """
  x,seqid,seqnum,tgt = data.getitem(idx)
  The getitem() function inputs an index between 0 and len(data)-1
  and returns a data example corresponding to a sequence (seqid or seqnum)
  and fly (tgt) corresponding to input idx.
  The output data x is reordered so that the selected fly is first, and
  other flies are ordered by distance to that fly.
  Input:
  idx: Integer between 0 and len(self)-1 specifying which sample to select.
  Output:
  x is an ndarray of size seql x maxnflies x nkeypoints x 2 data sample
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
    d = self.get_fly_dists(self.X[seqid], tgt)
    order = np.argsort(d)
    x = self.X[seqid][:,order,...]
    return x,seqid,seqnum,tgt
  
  """
  x = data[idx]
  The __getitem__() function inputs an index between 0 and len(data)-1
  and returns a data example corresponding to a sequence (seqid or seqnum)
  and fly (tgt) corresponding to input idx.
  The output data x is reordered so that the selected fly is first, and
  other flies are ordered by distance to that fly.
  Input:
  idx: Integer between 0 and len(self)-1 specifying which sample to select.
  Output:
  x is an ndarray of size seql x maxnflies x nkeypoints x 2 data sample
  """
  def __getitem__(self,idx):
    x,_,_,_ = self.getitem(idx)
    return x
  
  """
  d = data.get_fly_dists(x,tgt=0)
  Compute the distance between fly tgt and all other flies. This is defined as the
  minimum distance between any pair of the following keypoints:
  'antennae','end_notum_x','end_abdomen'
  at middle frame data.ctrf
  Input:
  x: ndarray of size seql x maxnflies x nkeypoints x 2 data sample, sequence of data for all flies
  tgt: (optional) which fly to compute distances to. Default is 0.
  Output:
  d: Array of length nflies with the squared distance to the selected target.
  """
  def get_fly_dists(self, x, tgt=0):
    nkpts = len(self.fidxdist)
    d = np.min(np.sum((x[self.ctrf,:,self.fidxdist,:].reshape((nkpts,1,self.ntgts,self.d))-
                x[self.ctrf,tgt,self.fidxdist,:].reshape((1,nkpts,1,self.d)))**2.,axis=3),
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
  poses: Required. nflies x nfeatures x 2 ndarray.
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
      x,seqid,seqnum,tgt = self[idx]
      return plot_flies(x[f, :, :, :], self.keypointidx, self.skeleton_edges, **kwargs)
    
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
x: ndarray of arbitrary dimensions, as long as the last two dimensions are nfeatures x 2,
and correspond to the keypoints and x,y coordinates.
"""
def get_real_flies(x):
  # x is ntgts x nfeatures x 2
  isreal = np.all(np.isnan(x),axis=(-1,-2))==False
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
poses: Required. nflies x nfeatures x 2 ndarray.
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
  nflies = poses.shape[0]
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
    hkpts[fly],hedges[fly],fig,ax = plot_fly(poses[fly,...],fig=fig,ax=ax,color=colors[fly,...],
                                             kptcolors=kptcolors,hedge=hedges[fly],hkpt=hkpts[fly],**kwargs)
  if isnewaxis:
    ax.axis('equal')
  
  return hkpts,hedges,fig,ax

"""
animate_pose_sequence(seq=None, kptidx=None, skelidx=None,
                      start_frame=0,stop_frame=None,skip=1,
                      fig=None,ax=None,savefile=None,
                      **kwargs)
Visualize all flies for the input sequence of frames seq.
Inputs:
seq: Required. seql x nflies x nfeatures x 2 ndarray.
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
    stop_frame = seq.shape[0]
  fig,ax,isnewaxis = set_fig_ax(fig=fig,ax=ax)
  
  isreal = get_real_flies(seq)
  idxreal = np.where(np.any(isreal,axis=0))[0]
  seq = seq[:,idxreal,...]

  # plot the arena wall
  theta = np.linspace(0,2*np.pi,360)
  ax.plot(ARENA_RADIUS_MM*np.cos(theta),ARENA_RADIUS_MM*np.sin(theta),'k-',zorder=-10)
  minv = -ARENA_RADIUS_MM*1.01
  maxv = ARENA_RADIUS_MM*1.01
  
  # first frame
  f = start_frame
  h = {}
  h['kpts'],h['edges'],fig,ax = plot_flies(poses=seq[f,...],fig=fig,ax=ax,**kwargs)
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
    h['kpts'],h['edges'],fig,ax = plot_flies(poses=seq[f,...],fig=fig,ax=ax,
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
  #hkpts,hedges,fig,ax = datatrain.plot_flies(x[datatrain.ctrf, :, :, :])
  
  # Set savefile if we want to save a video
  #savefile = 'seq.gif'
  savefile=None
  
  # get a different sequence in a different way
  x = datatrain[234]
  # plot that sequence
  animate_pose_sequence(seq=x,kptidx=datatrain.keypointidx,skelidx=datatrain.skeleton_edges,savefile=savefile)

if __name__ == "__main__":
  DatasetTest()
  