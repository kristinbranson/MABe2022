%% read in data locations
rootdatadir = 'sharedata20211226';
splitinfofile = 'SplitInfo.csv';
datafilestr = 'data.mat';
% splitnames are
% usertrain
% testtrain
% test1
% test2

load PrepareDataState20211223 Xnames ynames
Xnames0 = Xnames;
ynames0 = ynames;
nfeaturesX0 = numel(Xnames0);
nfeaturesy0 = numel(ynames0);

splitinfo = importdata(splitinfofile);
expnames = splitinfo.textdata(2:end,1);
exp2splitidx = splitinfo.data;
exp2splitname = splitinfo.textdata(2:end,2);
assert(numel(exp2splitidx)==numel(expnames));
expdirs = cellfun(@(x) fullfile(rootdatadir,x),expnames,'Uni',0);
assert(all(cellfun(@(x) exist(x,'dir'),expdirs)>0));
nexps = numel(expdirs);

nsplits = max(exp2splitidx);
splittypes = cell(1,nsplits);
nexpspersplit = nan(1,nsplits);
for i = 1:nsplits,
  splittypes{i} = exp2splitname{find(exp2splitidx==i,1)};
  nexpspersplit(i) = nnz(exp2splitidx==i);
end

% how big will each split be? 
expi = 1;
load(fullfile(expdirs{expi},datafilestr),'X');
sizeinfo = whos('X');
ngigspersplit = sizeinfo.bytes*nexpspersplit/2^30;
for spliti = 1:nsplits,
  fprintf('Approx amount of space to hold %s split X: %.1f GB\n',splittypes{spliti},ngigspersplit(spliti));
end

%% create big files with each split

% reseed the random number generator so that this code always produces the
% same results
rng('default');

reorderinfo = struct;
reorderinfo.exps = randperm(nexps);
reorderinfo.idsperexp = cell(1,nexps);
% reorderinfo.fliesperexp uses the original experiment index

% fidx = struct;
% fidx.frame = 1;
% fidx.fly = 2;
% fidx.video = 3;
% nfeaturesadd = numel(fieldnames(fidx));
Xnames = Xnames0;
% Xnames{fidx.frame} = 'frame';
% Xnames{fidx.fly} = 'fly';
% Xnames{fidx.video} = 'video';
ynames = ynames0;
nfeaturesX = nfeaturesX0;
nfeaturesy = nfeaturesy0;

fnamespos = {...
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
  };
  
[ism,fidxpos] = ismember(fnamespos,Xnames);
assert(all(ism));

maxnflies = 0;
nfliespersplit = nan(1,nsplits);

for spliti = 1:nsplits,
  
  fprintf('Split %d (%s)\n',spliti,splittypes{spliti});
    
  dataoff = 0;
  idoff = 1;
  videooff = 1;
  
  X = nan(nfeaturesX,maxnflies,0);
  y = nan(nfeaturesy,maxnflies,0);
  frames = nan(1,0);
  ids = nan(maxnflies,0);
  videoidx = nan(1,0);
  
  for expii = 1:nexps,
    expi = reorderinfo.exps(expii);
    if exp2splitidx(expi)~=spliti,
      continue;
    end
    fprintf('Exp %d / %d\n',videooff,nexpspersplit(spliti));
    dcurr = load(fullfile(expdirs{expi},datafilestr));
    [ntrajs,nframes,~] = size(dcurr.X);
    
    sfs = nan(1,ntrajs);
    efs = nan(1,ntrajs);
    for fly = 1:ntrajs,
      sf = 1;
      ef = nframes;
      ismissing = all(isnan(dcurr.X(fly,:,:)),3);
      sf1 = find(ismissing(1:end-1)&~ismissing(2:end));
      assert(numel(sf1)<=1);
      if ~isempty(sf1),
        sf = sf1+1;
      end
      ef1 = find(ismissing(2:end)&~ismissing(1:end-1));
      assert(numel(ef1)<=1);
      if ~isempty(ef1),
        ef = ef1;
      end
      sfs(fly) = sf;
      efs(fly) = ef;
    end
    pos = permute(dcurr.X(:,:,fidxpos),[3,1,2]);
    [assignment,cost] = PackTrajs(sfs,efs,pos);

    reorderids = randperm(ntrajs)+idoff;
    reorderinfo.idsperexp{expii} = reorderids;
    
    nfliescurr = max(max(assignment),maxnflies);
    Xpack = nan(nfeaturesX,nfliescurr,nframes);
    ypack = nan(nfeaturesy,nfliescurr,nframes);
    %Xpack(fidx.frame,:,:) = repmat(1:nframes,[nfliescurr,1]);
    %Xpack(fidx.video,:,:) = videooff;
    idspack = nan(nfliescurr,nframes);
  
    for id = 1:ntrajs,
      flycurr = assignment(id);
      assert(all(isnan(idspack(flycurr,sfs(id):efs(id)))));
      Xpack(1:size(dcurr.X,3),flycurr,sfs(id):efs(id)) = permute(dcurr.X(id,sfs(id):efs(id),:),[3,1,2]);
      %Xpack(fidx.fly,flycurr,sfs(id):efs(id)) = reorderids(id);
      idspack(flycurr,sfs(id):efs(id)) = reorderids(id);
      ypack(:,flycurr,sfs(id):efs(id)) = permute(dcurr.y(id,sfs(id):efs(id),:),[3,1,2]);
    end

    firstid = nan(1,nfliescurr);
    for fly = 1:nfliescurr,
      if fly > max(assignment),
        firstid(fly) = nan;
      else
        firstid(fly) = idspack(fly,find(~isnan(idspack(fly,:)),1));
      end
    end
    [~,roworder] = sort(firstid);
    Xpack = Xpack(:,roworder,:);
    idspack = idspack(roworder,:);
    ypack = ypack(:,roworder,:);
    if nfliescurr > maxnflies,
      X(:,maxnflies+1:nfliescurr,:) = nan;
      y(:,maxnflies+1:nfliescurr,:) = nan;
      ids(maxnflies+1:nfliescurr,:) = nan;
      maxnflies = nfliescurr;
    end
    X(:,:,dataoff+1:dataoff+nframes) = Xpack;
    y(:,:,dataoff+1:dataoff+nframes) = ypack;
    ids(:,dataoff+1:dataoff+nframes) = idspack;
    frames(dataoff+1:dataoff+nframes) = 1:nframes;
    videoidx(dataoff+1:dataoff+nframes) = videooff;
    
    idoff = idoff + ntrajs;
    dataoff = dataoff + nframes;
    videooff = videooff+1;
  end
  nfliespersplit(spliti) = maxnflies;
  
  save(sprintf('X%s.mat',splittypes{spliti}),'X','frames','ids','videoidx');
  save(sprintf('y%s.mat',splittypes{spliti}),'y','frames','ids','videoidx');
  
end

fid = fopen('FeatureNames.txt','w');
fprintf(fid,'%s\n',Xnames{:});
fclose(fid);

save ReorderExpsVideosInfo.mat reorderinfo

%% Make all the splits the same size

for spliti = 1:nsplits,
  
  if nfliespersplit(spliti) == maxnflies,
    continue;
  end
    
  tmp = load(sprintf('X%s.mat',splittypes{spliti}));
  tmp.X(:,nfliespersplit(spliti)+1:maxnflies,:) = nan;
  tmp.ids(nfliespersplit(spliti)+1:maxnflies,:) = nan;  
  save(sprintf('X%s.mat',splittypes{spliti}),'-struct','tmp');
  tmp = load(sprintf('y%s.mat',splittypes{spliti}),'y');
  tmp.y(:,nfliespersplit(spliti)+1:maxnflies,:) = nan;
  save(sprintf('y%s.mat',splittypes{spliti}),'-struct','tmp');
  
end
clear tmp;

%% spot sanity check

expi = 37;
expii = find(reorderinfo.exps==expi);
spliti = splitidx(expi);
load(sprintf('X%s.mat',splittypes{spliti}));
load(sprintf('y%s.mat',splittypes{spliti}));
dcurr = load(fullfile(expdirs{expi},datafilestr));
ntrajs = size(dcurr.X,1);
isdata = ~all(isnan(dcurr.X),3);
for traji = 1:ntrajs,
  id = reorderinfo.idsperexp{expii}(traji);
  idxcurr = ids == id;
  % make sure only one row
  isid = any(idxcurr,2);
  assert(nnz(isid)==1,'id assigned to more than one fly');
  idxcurr = idxcurr(isid,:);
  assert(nnz(idxcurr) == nnz(isdata(traji,:)),'number of frames of data mismatch');
  Xnew = permute(X(:,isid,idxcurr),[1,3,2]);
  framesnew = frames(idxcurr);
  
  sf = find(isdata(traji,:),1);
  ef = find(isdata(traji,:),1,'last');
  Xbase = permute(dcurr.X(traji,sf:ef,:),[3,2,1]);
  framesbase = sf:ef;
  assert(isequaln(Xnew,Xbase),'data mismatch');
  assert(isequaln(framesnew,framesbase),'frame mismatch');
end

%% raw data - no temporal or social information

ynamesordinal = {
  'light_strength'
  'light_period'
  };

testdata = load('Xtest1.mat');
tmp = load('ytest1.mat');
testdata.y = tmp.y;
[testdata.X,featurenames] = BasicCleanXFeatures(testdata.X,Xnames);

traindata = load('Xtesttrain.mat');
tmp = load('ytesttrain.mat');
traindata.y = tmp.y;
traindata.X = BasicCleanXFeatures(traindata.X,Xnames);
clear tmp;

traindatacollapse = CollapseData_1Animal1Frame(traindata);
testdatacollapse = CollapseData_1Animal1Frame(testdata);

% compute mean and standard deviation of training data
trainmu = nanmean(traindatacollapse.X,2);
trainsig = nanmean(traindatacollapse.X,2);

%% train a logistic regression classifier

maxntrain = 10000;

for yi = 1:numel(ynames),
  
  yname = ynames{yi};
  isordinal = ismember(yname,ynamesordinal);
  
  fprintf('%s:\n',yname);
  
  [Xtrain,ytrain] = BasicCleanXy(traindatacollapse.X,traindatacollapse.y(yi,:),trainmu,trainsig);
  labelscurr = unique(ytrain(:));
  nlabelscurr = numel(labelscurr);
  if ~isordinal,
    assert(nlabelscurr==2);
  end
  if isordinal,
    continue;
  end
      
  % sample some data for training, use the rest for holdout
  counts = hist(ytrain,labelscurr);
  ncurr = numel(ytrain);
  nsample = min(maxntrain,ncurr);
  psample = min(1,nsample./counts/nlabelscurr);
  dosample = rand(ncurr,1);
  for labeli = 1:nlabelscurr,
    label = labelscur(labeli);
    dosample(ytrain==label) = dosample(ytrain==label) <= psample(labeli);
  end
  dosample = dosample ~= 0;
  nsamplecurr = nnz(dosample);
  Xsample = Xtrain(:,dosample);
  ysample = ytrain(dosample);
  
  [~,ysampleidx] = ismember(ysample,labelscurr);
  
  [coeffs,dev,stats] = mnrfit(Xsample',ysampleidx');
  [~,order] = sort(abs(coeffs(2:end)),'descend');
  for ii = 1:3,
    i = order(ii);
    fprintf('%s: coeff = %f, p = %f\n',featurenames{i},coeffs(i+1),stats.p(i+1));
  end
  
  ysampleidxfit = mnrval(coeffs,Xsample');
  ysampleidxfit = max(ysampleidxfit,[],2)';
  %ysamplefit = (ysamplefit(:,2) > ysamplefit(:,1))';
  trainconfusionmatrx = ComputeConfusionMatrix(ysampleidx,ysampleidxfit,1:nlabelscurr);
  for labeli = 1:nlabelscurr,
    label = labelscurr(labeli);
    s = sum(trainconfusionmatrix(labeli,:));
    fprintf('Train label = %d, error fraction = %d / %d = %f\n',label,s-trainconfusionmatrix(labeli,labeli),s,(s-trainconfusionmatrix(labeli,labeli))/s);
  end

  [Xtest,ytest] = BasicCleanXy(testdatacollapse.X,testdatacollapse.y(yi,:),trainmu,trainsig);
  ytestfit = mnrval(coeffs,Xtest');
  ytestfit = (ytestfit(:,2) > ytestfit(:,1))';
  testconfusionmatrix = ComputeConfusionMatrix(ytest,ytestfit,[0,1]);
  fprintf('Test n. false positives = %d / %d = %f\n',testconfusionmatrix(1,2),sum(testconfusionmatrix(1,:)),testconfusionmatrix(1,2)/sum(testconfusionmatrix(1,:)));
  fprintf('Test n. false negatives = %d / %d = %f\n',testconfusionmatrix(2,1),sum(testconfusionmatrix(2,:)),testconfusionmatrix(2,1)/sum(testconfusionmatrix(2,:)));

end
