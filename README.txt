
=================================

==MABeFlyUtils.py==

Useful functions for interacting with the files X<split>_seq.npy.

FlyDataset is a pytorch Dataset for loading and grabbing frames.
plot_fly(), plot_flies(), and animate_pose_sequence()
can be used to visualize a fly, a frame, and a sequence of frames. 

=================================

==SaveFlyData.py==

Convert from MATLAB data for videos to numpy data for sequences for
each of the following data splits (<split> below):
usertrain: Data that the competitors can use to train their embeddings.
testtrain: Data that will be used to train the simple (linear) classifier
from the embedding.
test1: Data that will be used to test the simple classifier. This will
be used in the submission process, and competitors can submit multiple
times.
test2: Data that will be used to test the simple classifier for the
final competition. Competitors will not get feedback on this data. 

Competitors should be given access to all X data throughout the competition.
ytesttrain and ytest1 will be used during initial submission queries
to the competition, and ytest2 will only be used once at the end. 

=Front-facing functions=

resave() goes from Matlab to npz files (my first stab at saving the data).
This creates X<split>.npz and y<split>.npz files for each split type.

X<split>.npz contains ndarray X, videoidx, ids, and frames.
These are all created by concatenated over axis=0 data across all videos
in this split. X is nframestotal x maxnflies x nfeatures.
nframestotal is the total number of frames of data we have across all
videos.
maxnflies is the maximum number of flies tracked in any frame.
nfeatures is the number of features we have to describe the position of
each fly (includes both x- and y-coordinates of keypoints).
X will be nan when there is no data, e.g. if we have only tracked 9 flies
in a given frame, even though maxnflies=11. 
Which videos, fly identities/trajectories, and frames each X[i,j,:,:]
corresponds to is stored in
videoidx (nframestotal x 1)
ids (nframestotal x maxnflies)
frames (nframestotal x 1)
Note that within this X matrix, X[i,j,:,:] might correspond
to different videos and fly identities. The following features are stored,
in this order:
[
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

y<split>.npz contains ndarray y, videoidx, ids, and frames.
These are all created by concatenated over axis=0 data across all videos
in this split. y is nframestotal x maxnflies x nclasses. 
Which videos, fly identities/trajectories, and frames each y[i,j,:,:]
corresponds to is stored in
videoidx (nframestotal x 1)
ids (nframestotal x maxnflies)
frames (nframestotal x 1)
(as in X<split>.npz).
Unique values in y are 0,1, and nan, with 0 and 1 meaning negative and
positive labels, and nan meaning unlabeled.

CutAllIntoSeqs() goes from the npz files saved by resave() to sequences
similar to the format for the mouse data. It saves X<split>_seq.npy,
y<split>_seq.npy, and y<split>_record.npy for each of the data splits.

X<split>_seq.npy contains data for contiguous sequences of
seql = 4500 frames = 30 seconds. keypoints is a dictionary indexed by
sequence ids, which are 20-character random strings.
keypoints[seqid] is an ndarray of size seql x maxnflies x nkeypoints x 2.
maxnflies is the maximum number of flies tracked in any frame of any
video. 
nkeypoints is the number of features we have to describe the position of
each fly. The features stored in keypoints[seqid][...,i,j] are:
[('wing_left_x', 'wing_left_y'),
 ('wing_right_x', 'wing_right_y'),
 ('antennae_x_mm', 'antennae_y_mm'),
 ('right_eye_x_mm', 'right_eye_y_mm'),
 ('left_eye_x_mm', 'left_eye_y_mm'),
 ('left_shoulder_x_mm', 'left_shoulder_y_mm'),
 ('right_shoulder_x_mm', 'right_shoulder_y_mm'),
 ('end_notum_x_mm', 'end_notum_y_mm'),
 ('end_abdomen_x_mm', 'end_abdomen_y_mm'),
 ('middle_left_b_x_mm', 'middle_left_b_y_mm'),
 ('middle_left_e_x_mm', 'middle_left_e_y_mm'),
 ('middle_right_b_x_mm', 'middle_right_b_y_mm'),
 ('middle_right_e_x_mm', 'middle_right_e_y_mm'),
 ('tip_front_right_x_mm', 'tip_front_right_y_mm'),
 ('tip_middle_right_x_mm', 'tip_middle_right_y_mm'),
 ('tip_back_right_x_mm', 'tip_back_right_y_mm'),
 ('tip_back_left_x_mm', 'tip_back_left_y_mm'),
 ('tip_middle_left_x_mm', 'tip_middle_left_y_mm'),
 ('tip_front_left_x_mm', 'tip_front_left_y_mm'),
 ('x_mm', 'y_mm'),
 ('cos_ori', 'sin_ori'),
 ('maj_ax_mm', 'min_ax_mm'),
 ('body_area_mm2', 'fg_area_mm2'),
 ('img_contrast', 'min_fg_dist')]
These feature names are stored in the vocabulary list. Note that for
tracked keypoints, keypoints[seqid][...,i,0] and
keypoints[seqid][...,i,1] are the (x,y) coordinates of that keypoint.
Other features are outputs from the Caltech FlyTracker, and have been
paired somewhat arbitrarily. keypoints[seqid] will be nan when there is
no data, e.g. if we have only tracked 9 flies in a given frame, even
though maxnflies=11. The order of flies is randomly shuffled for
each sequence, but keypoints[seqid][:,fly,:,:] will correspond to the
same fly throughout the sequence. Between sequences, I skip a random
amount of frames between .5 and 2 seconds. Before all of this
randomization, I seed the random number generator with a fixed
value so that we can repeat this analysis. 

y<split>_seq.npy contains data corresponding to X<split>_seq.npy.
annotations is a dictionary indexed by the same sequence identifying
strings as keypoints in X<split>_seq.npy.
annotations[seqid] is nclasses x seql x maxnflies.
nclasses is the number of classification tasks we have defined (50).
Unique values in annotations[seqid] are 0,1, and nan, with 0 and 1
meaning negative and positive labels, and nan meaning unlabeled.
The names of the categorization tasks are stored in
the vocabulary list. 
