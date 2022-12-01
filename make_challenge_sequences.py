import numpy as np
import glob, os
import copy
import random
import string
from scipy.ndimage.filters import maximum_filter1d


basedir = 'X:\\JAX\\'
pthlist = ['X:\\JAX\\B6J_3M_stranger_4day\\', 'X:\\JAX\\BTBR_3M_stranger_4day\\']


def move_max(data,m1,m2):
    # what is the furthest distance a valid keypoint (minus tail) moved on each frame?
    
    dt_mask = (data[1:,:,:10,:]==0) | (data[:-1,:,:10,:]==0)
    masked_data = np.ma.masked_where(dt_mask, data[1:,:,:10,:])
    
    dt = np.square(masked_data[1:,m1] - masked_data[:-1,m2])
    ss = np.max(np.sum(dt, axis=2), axis=1)
    vel = np.sqrt(ss).data
    return vel


def group_extent(data):
    # what is the extent of the 3-mouse group on each frame? used for identifying sequences that are huddling only
    
    masked_data = np.ma.masked_where(data==0, data)
    
    group_extent = (np.max(masked_data[:,:,:,0], axis=(1,2)) - np.min(masked_data[:,:,:,0], axis=(1,2))) + \
                    (np.max(masked_data[:,:,:,1], axis=(1,2)) - np.min(masked_data[:,:,:,1], axis=(1,2)))
    return group_extent


def sequence_extent(data, seq_length=1800):
    # how much huddling is there in the next seq_length frames? IS IT TOO MUCH? GET A ROOM MICE
    framewise_extent = group_extent(data)
    seq_extent = maximum_filter1d(framewise_extent, size=seq_length) # valid are 1/2W:-1/2W
    seq_extent = np.concatenate((seq_extent[round(seq_length/2):], seq_extent[-round(seq_length/2):]))
    return seq_extent


def find_valid_sequences(data, seq_length=1800, num_sequences=6):
    dt = np.zeros((3,len(data)-2))

    for mouse in range(3):
        dt[mouse,:] = move_max(data,mouse,mouse)

    jumpy_frames = np.where(np.max(dt,axis=0) > 400)
    if jumpy_frames:
        jumpy_frames = np.concatenate((np.zeros((1)), jumpy_frames[0], np.array([np.shape(data)[0]])))
        stable_durations = jumpy_frames[1:] - jumpy_frames[:-1]
    valid_sequences = np.where(stable_durations > seq_length)

    valid_starts = []
    valid_stops = []
    if valid_sequences:
        for s in valid_sequences[0]:
            valid_starts.append(jumpy_frames[s].astype(int))
            valid_stops.append(jumpy_frames[s+1].astype(int))
    
    # generate a list of all possible valid start times
    start_times = []
    for i,j in zip(valid_starts, valid_stops):
        start_times += list(range(i, j-seq_length-1))
    print('  ' + str(int(len(start_times)/len(dt[0])*100)) + '% valid frames')
    if not start_times:
        return []
    
    # pick <num_sequences> of them uniformly from the list (hopefully that's good enough??)
    return [start_times[i] for i in np.linspace(0, len(start_times)-1, num_sequences).astype(int)]


seq_length = 30 * 60
num_starts = 6
fr_huddle = 0.2 # only keep a fraction of sequences that are all-huddle

# seq_starts = [0, 10, 20, 30, 40, 50]
batchsize = 250

all_data = {}
count = 0
nfiles = 0
for pth in pthlist:
    files = glob.glob(pth + '*.npy')
    random.shuffle(files)
    
    for f in files:
        count += 1
        print(str(count).zfill(4) + ': ' + os.path.basename(f))
        if not count % batchsize:
            print('writing file ' + str(nfiles))
            
            # get rid of key ordering since it's correlated with labels
            order_keys = sorted(all_data['sequences'])
            data_shuff = {'vocabulary': all_data['vocabulary'], 'sequences':{}}
            for key in order_keys:
                data_shuff['sequences'][key] = all_data['sequences'][key]
            
            np.save(os.path.join(basedir, 'all_data_' + str(nfiles).zfill(3) + '.npy'), data_shuff)
            
        data = np.load(f, allow_pickle=True).item()
        if not all_data:
            all_data['vocabulary'] = data['vocabulary']
            all_data['sequences'] = {}
        # convert start hour to start minute
        data['annotations'][1,:] = data['annotations'][1,:]*60
        
        # find all seq_length-long snippets of time that don't have dramatic position jumps
        # we double the number of start times to pull because some of them may be huddles that we'll want to drop
        seq_starts = find_valid_sequences(data['keypoints'], seq_length=seq_length, num_sequences=num_starts*2)
        if not seq_starts:  # no valid sequences in the file? hopefully this doesn't happen
            print('  no valid sequences found')
            continue
        random.shuffle(seq_starts) # keep the first <num_starts> that pass the huddle test
        
        used_starts = 0
        for start_time in seq_starts:
            if used_starts >= num_starts:
                continue
            end_time = start_time + seq_length
            data_sub = {}
            data_sub['annotations']      = (data['annotations'][:,start_time:end_time]).astype(int)
            data_sub['annotations'][1,:] = data_sub['annotations'][1,:] + round(start_time/30/60)
            data_sub['keypoints']        = data['keypoints'][start_time:end_time,:,:,:].astype(int)
            data_sub['confidence']  = data['confidence'][start_time:end_time,:,:]
            
            if np.max(group_extent(data_sub['keypoints'])) < 400:  # guys stop huddling >:[
                if np.random.uniform() > fr_huddle:
                    continue
            used_starts += 1
            res = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 20))
            all_data['sequences'][res] = data_sub

# get rid of key ordering for the last batch
order_keys = sorted(all_data['sequences'])
data_shuff = {'vocabulary': all_data['vocabulary'], 'sequences':{}}
for key in order_keys:
    data_shuff['sequences'][key] = all_data['sequences'][key]

np.save(os.path.join(basedir, 'all_data_' + str(nfiles).zfill(3) + '.npy'), data_shuff)