function [X,featurenames] = BasicCleanXFeatures(X,Xnames)

% only keep some features, replace angle with cos and sin
featuresuse = {
  'x_mm'
  'y_mm'
  'maj_ax_mm'
  'min_ax_mm'
  'wing_left_x'
  'wing_left_y'
  'wing_right_x'
  'wing_right_y'
  'body_area_mm2'
  'fg_area_mm2'
  'img_contrast'
  'min_fg_dist'
  'antennae_x_mm'
  'antennae_y_mm'
  'right_eye_x_mm'
  'right_eye_y_mm'
  'left_eye_x_mm'
  'left_eye_y_mm'
  'left_shoulder_x_mm'
  'left_shoulder_y_mm'
  'right_shoulder_x_mm'
  'right_shoulder_y_mm'
  'end_notum_x_mm'
  'end_notum_y_mm'
  'end_abdomen_x_mm'
  'end_abdomen_y_mm'
  'middle_left_b_x_mm'
  'middle_left_b_y_mm'
  'middle_left_e_x_mm'
  'middle_left_e_y_mm'
  'middle_right_b_x_mm'
  'middle_right_b_y_mm'
  'middle_right_e_x_mm'
  'middle_right_e_y_mm'
  'tip_front_right_x_mm'
  'tip_front_right_y_mm'
  'tip_middle_right_x_mm'
  'tip_middle_right_y_mm'
  'tip_back_right_x_mm'
  'tip_back_right_y_mm'
  'tip_back_left_x_mm'
  'tip_back_left_y_mm'
  'tip_middle_left_x_mm'
  'tip_middle_left_y_mm'
  'tip_front_left_x_mm'
  'tip_front_left_y_mm'
  };
oriidx = find(strcmpi(Xnames,'ori_rad'));
assert(numel(oriidx)==1);
useidx = ismember(Xnames,featuresuse);

ori = X(oriidx,:,:);
X = cat(1,X(useidx,:,:),cos(ori),sin(ori));
featurenames = cat(2,Xnames(useidx),{'cos_ori','sin_ori'});
