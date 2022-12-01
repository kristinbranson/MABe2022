import glob, os
import yaml
import numpy as np
import copy
from datetime import datetime
# import matplotlib.pyplot as plt
import h5py


pthlist = ['X:\\JAX\\B6J_3M_stranger_4day\\', 'X:\\JAX\\BTBR_3M_stranger_4day\\']

for pth in pthlist:
    stats_files = glob.glob(pth + '**\\**\\results.yaml')
    strain = int('BTBR' in pth)

    for stats_file in stats_files:
        experiment = os.path.dirname(stats_file)
        print(experiment.replace(pth,'') + ' --------------------------')
        expt_name = experiment.replace(pth, '').replace('\\', '_')

        data = {}
        with open(os.path.join(stats_file), 'r') as fp:
            temp = yaml.safe_load_all(fp)
            for d in temp:
                data[list(d.keys())[0]] = d[list(d.keys())[0]]

        files = list(data.keys())

        behaviors = list(data[files[0]].keys())
        behaviors.remove('frame_count')

        vocab = ['experiment_day','time_of_day','strain']
        [vocab.append(b) for b in behaviors]

        dt0 = datetime.strptime(files[0][files[0].find('_')+1:], '%Y-%m-%d_%H-%M-%S')
        day0 = dt0.month*30 + dt0.day

        for file in files[1:]:
            print(file)
            dt = datetime.strptime(file[file.find('_')+1:], '%Y-%m-%d_%H-%M-%S')
            file_day = dt.month*30 + dt.day - day0
            file_start = dt.hour
            timestamps = file_start*60*60*30 + np.linspace(0, 60*60*30 - 1, data[file]['frame_count'], dtype=int)

            data_file_name = os.path.join(experiment, file + '_pose_est_v3_fixIDs.h5')
            if not os.path.exists(data_file_name):
                print('I couldn''t find ' + data_file_name)
                continue

            with h5py.File(data_file_name, 'r') as pose_h5:
                vid_grp = pose_h5['poseest']
                all_points = vid_grp['points'][:]
                all_confidence = vid_grp['confidence'][:]

            M = np.zeros((len(vocab),data[file]['frame_count']))
            M[0,:] = np.full((data[file]['frame_count']), file_day)
            M[1,:] = np.full((data[file]['frame_count']), timestamps)
            M[2,:] = np.full((data[file]['frame_count']), strain)
            for i, b in enumerate(behaviors):
                for bout in data[file][b]:
                    M[i+3,bout['start_frame']:bout['stop_frame_exclu']] = 1

            np.save(os.path.join(pth, expt_name + '_' + file + '_compiled.npy'), {'vocabulary': vocab,
                                                                                  'annotations': M,
                                                                                  'keypoints': all_points,
                                                                                  'confidence': all_confidence})
