import numpy as np
import os
import re
import h5py
import random
import string

SEED = 19700
FPS = 150
datadir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022'

xusertrainfile = os.path.join(datadir, 'Xusertrain.npz')
xtesttrainfile = os.path.join(datadir, 'Xtesttrain.npz')
xtest1file = os.path.join(datadir, 'Xtest1.npz')
xtest2file = os.path.join(datadir, 'Xtest2.npz')
yusertrainfile = os.path.join(datadir, 'yusertrain.npz')
ytesttrainfile = os.path.join(datadir, 'ytesttrain.npz')
ytest1file = os.path.join(datadir, 'ytest1.npz')
ytest2file = os.path.join(datadir, 'ytest2.npz')

Xnames = [
  'x_mm',
  'y_mm',
  'cos_ori',
  'sin_ori',
  'maj_ax_mm',
  'min_ax_mm',
  'wing_left_x',
  'wing_left_y',
  'wing_right_x',
  'wing_right_y',
  'body_area_mm2',
  'fg_area_mm2',
  'img_contrast',
  'min_fg_dist',
  'antennae_x_mm',
  'antennae_y_mm',
  'right_eye_x_mm',
  'right_eye_y_mm',
  'left_eye_x_mm',
  'left_eye_y_mm',
  'left_shoulder_x_mm',
  'left_shoulder_y_mm',
  'right_shoulder_x_mm',
  'right_shoulder_y_mm',
  'end_notum_x_mm',
  'end_notum_y_mm',
  'end_abdomen_x_mm',
  'end_abdomen_y_mm',
  'middle_left_b_x_mm',
  'middle_left_b_y_mm',
  'middle_left_e_x_mm',
  'middle_left_e_y_mm',
  'middle_right_b_x_mm',
  'middle_right_b_y_mm',
  'middle_right_e_x_mm',
  'middle_right_e_y_mm',
  'tip_front_right_x_mm',
  'tip_front_right_y_mm',
  'tip_middle_right_x_mm',
  'tip_middle_right_y_mm',
  'tip_back_right_x_mm',
  'tip_back_right_y_mm',
  'tip_back_left_x_mm',
  'tip_back_left_y_mm',
  'tip_middle_left_x_mm',
  'tip_middle_left_y_mm',
  'tip_front_left_x_mm',
  'tip_front_left_y_mm'
]
ynames = [
  'female',
  'control',
  'BDP_sexseparated',
  'Control_RGB',
  '71G01',
  'male71G01_femaleBDP',
  '65F12',
  '91B01',
  'BlindControl',
  'aIPgpublished3_newstim',
  'pC1dpublished1_newstim',
  'aIPgBlind_newstim',
  'BlindControl_offvson',
  'BlindControl_offvsstrong',
  'BlindControl_offvsweak',
  'BlindControl_weakvsstrong',
  'BlindControl_firstvslast_strong',
  'Control_RGB_offvson',
  'Control_RGB_offvsstrong',
  'Control_RGB_offvsweak',
  'Control_RGB_weakvsstrong',
  'Control_RGB_firstvslast_strong',
  'aIPgBlind_newstim_offvson',
  'aIPgBlind_newstim_offvsstrong',
  'aIPgBlind_newstim_offvsweak',
  'aIPgBlind_newstim_weakvsstrong',
  'aIPgBlind_newstim_firstvslast_strong',
  'aIPgpublished3_newstim_offvson',
  'aIPgpublished3_newstim_offvsstrong',
  'aIPgpublished3_newstim_offvsweak',
  'aIPgpublished3_newstim_weakvsstrong',
  'aIPgpublished3_newstim_firstvslast_strong',
  'pC1dpublished1_newstim_offvson',
  'pC1dpublished1_newstim_offvsstrong',
  'pC1dpublished1_newstim_offvsweak',
  'pC1dpublished1_newstim_weakvsstrong',
  'pC1dpublished1_newstim_firstvslast_strong',
  'courtship',
  'control',
  'blind',
  'aIPg',
  'aggression',
  'GMR71G01',
  'sexseparated',
  'perframe_aggression',
  'perframe_chase',
  'perframe_courtship',
  'perframe_highfence2',
  'perframe_wingext',
  'perframe_wingflick'
]



def loadmatdata(matfile):
  
  datacurr = h5py.File(matfile, 'r')
  if 'X' in datacurr.keys():
    X = np.array(datacurr['X'])
  elif 'y' in datacurr.keys():
    X = np.array(datacurr['y'])
  else:
    X = None
  if 'videoidx' in datacurr.keys():
    videoidx = np.array(datacurr['videoidx']).astype(int)
    videoidx -= 1
  else:
    videoidx = None
  if 'ids' in datacurr.keys():
    ids = np.array(datacurr['ids'])
    ids[np.isnan(ids)] = 0
    ids = ids.astype(int)
    ids -= 1
  else:
    ids = None
  if 'frames' in datacurr.keys():
    frames = np.array(datacurr['frames']).astype(int)
    frames -= 1
  else:
    frames = None
  return X,videoidx,ids,frames

def mat2npz(infile,outfile,type='X'):
  X,videoidx,ids,frames = loadmatdata(infile)
  if type == 'y':
    np.savez_compressed(outfile,videoidx=videoidx,ids=ids,frames=frames,y=X)
  else:
    np.savez_compressed(outfile,videoidx=videoidx,ids=ids,frames=frames,X=X)

def GenerateSeqID():
  id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 20))
  return id

def CreateSeq(keypoints):
  seq = {'keypoints',keypoints}
  return seq

def sampleresave(inxfile,inyfile,outxfile,outyfile,outrecordfile,seql=30*FPS,borderl=5*FPS,skipl=(FPS//2,2*FPS)):
  data = {}
  with np.load(inxfile) as data1:
    data['X'] = data1['X']
    data['videoidx'] = data1['videoidx']
    data['ids'] = data1['ids']
    data['frames'] = data1['frames']
  with np.load(inyfile) as data1:
    data['y'] = data1['y']
    
  nframes = data['y'].shape[0]
  nflies = data['y'].shape[1]
  nclasses = data['y'].shape[2]
  nfeatures = data['X'].shape[2]
  
  xcoords = np.where(np.array(list(map(lambda x: re.search('_x(_mm)?$',x) is not None,Xnames))))[0]
  ycoords = np.where(np.array(list(map(lambda x: re.search('_y(_mm)?$',x) is not None,Xnames))))[0]
  assert(len(xcoords)==len(ycoords))

  specialpairs = [
    ('x_mm','y_mm'),
    ('cos_ori','sin_ori'),
    ('maj_ax_mm','min_ax_mm'),
    ('body_area_mm2','fg_area_mm2'),
    ('img_contrast','min_fg_dist',)
    ]
  
  xspecialidx = np.array(list(map(lambda x: Xnames.index(x[0]),specialpairs)))
  yspecialidx = np.array(list(map(lambda x: Xnames.index(x[1]),specialpairs)))
  assert(len(xspecialidx)==len(yspecialidx))
  
  xidx = np.concatenate((xcoords,xspecialidx))
  yidx = np.concatenate((ycoords,yspecialidx))
  nkeypoints = len(xidx)
  
  featurenames = list(map(lambda x,y: (Xnames[x],Xnames[y]),xidx,yidx))
  keypoints = {}
  annotations = {}
  record = {}
  record['idx'] = []
  record['names'] = []
  record['startframes'] = []
  record['videoidx'] = []
  record['ids'] = []
  counts = np.zeros((nclasses,2))
  isbreak = np.logical_or((data['videoidx'][:-1] != data['videoidx'][1:]).flatten(),
                          np.all(data['ids'][:-1,:]==data['ids'][1:,:],axis=1)==False)
  idxbreak = np.where(isbreak)[0]
  idxbreak = np.append(idxbreak,nframes)
  for breaki in range(len(idxbreak)-1):
    i0 = idxbreak[breaki]
    maxi1 = idxbreak[breaki+1]
    nseqscurr = 0
    while True:
      i1 = i0 + seql
      if i1 > maxi1:
        print('%d seqs (%d frames) stored between index %d and %d (%f fraction of %d frames)'%(nseqscurr,
                                                                                               nseqscurr*(seql-2*borderl),
                                                                                               idxbreak[breaki],maxi1,
                                                                                               nseqscurr*(seql-2*borderl)/(maxi1-idxbreak[breaki]),
                                                                                               maxi1-idxbreak[breaki]))
        break
      #if not data['videoidx'][i0] == data['videoidx'][i1]:
      #  i0 = i1
      #  continue
      kps = np.zeros((seql,nflies,nkeypoints,2),dtype=np.float32)
      kps[:,:,:,0] = data['X'][i0:i1,:,xidx]
      kps[:,:,:,1] = data['X'][i0:i1,:,yidx]
      #ann = np.zeros((nclasses,seql,nflies),dtype=bool)
      ann = np.transpose(data['y'][i0:i1,:,:],[2,0,1]).astype(np.float32)
      idxfly = np.append(np.random.permutation(np.where(data['ids'][i0,:] >= 0)[0]),
                         np.where(data['ids'][i0,:] < 0))
      kps = kps[:,idxfly,:,:]
      ann = ann[:,:,idxfly]
      
      for v in range(2):
        counts[:,v] = counts[:,v] + np.sum(ann[:,borderl:-borderl,:]==v,axis=(1,2))
        
      seqid = GenerateSeqID()
      keypoints[seqid] = kps
      annotations[seqid] = ann
      record['idx'].append(i0)
      record['startframes'].append(data['frames'][i0])
      record['videoidx'].append(data['videoidx'][i0])
      record['ids'].append(data['ids'][i0,idxfly])
      record['names'].append(seqid)
      off = random.randint(skipl[0],skipl[1])
      i0 = i1+off
      nseqscurr+=1

  order_keys = sorted(record['names'])
  xshuff = {'vocabulary':featurenames,'keypoints':{}}
  yshuff = {'vocabulary':ynames,'annotations':{}}
  for key in order_keys:
    xshuff['keypoints'][key] = keypoints[key]
    yshuff['annotations'][key] = annotations[key]
  
  print('N. labels within middles of sequences:')
  for i in range(nclasses):
    print('%s: 0: %d, 1: %d'%(ynames[i],counts[i,0],counts[i,1]))
  
  np.save(outxfile,xshuff)
  np.save(outyfile,yshuff)
  record['labelcounts'] = counts
  np.save(outrecordfile,record)

def resave():
  """
  Resave mat files as .npz files
  """
  Xfiles = [xusertrainfile, xtesttrainfile, xtest1file, xtest2file]
  yfiles = [yusertrainfile, ytesttrainfile, ytest1file, ytest2file]
  for outfile in Xfiles:
    f,ext = os.path.splitext(outfile)
    infile = f+'.mat'
    print('%s -> %s'%(infile,outfile))
    mat2npz(infile,outfile,'X')
  for outfile in yfiles:
    f,ext = os.path.splitext(outfile)
    infile = f+'.mat'
    print('%s -> %s'%(infile,outfile))
    mat2npz(infile,outfile,'y')
    
def CutAllIntoSeqs():
  random.seed(SEED)
  Xfiles = [xtest2file, xusertrainfile, xtesttrainfile, xtest1file]
  yfiles = [ytest2file, yusertrainfile, ytesttrainfile, ytest1file]
  
  for inxfile,inyfile in zip(Xfiles,yfiles):
    f,ext = os.path.splitext(inxfile)
    outxfile = f+'_seq.npy'
    f,ext = os.path.splitext(inyfile)
    outyfile = f+'_seq.npy'
    recordfile = f+'_record.npy'
    print('%s...'%inxfile)
    sampleresave(inxfile,inyfile,outxfile,outyfile,recordfile)
  
if __name__ == "__main__":
  #resave()
  CutAllIntoSeqs()
  