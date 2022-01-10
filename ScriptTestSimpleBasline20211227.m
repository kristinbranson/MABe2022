%% read in data locations
rootdatadir = 'sharedata20211230';
splitinfofile = 'SplitInfo.csv';
datafilestr = 'data.mat';
% splitnames are
% usertrain
% testtrain
% test1
% test2

SetUpPaths;

load PrepareDataState20211230 Xnames ynames
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

%% how big will each split be? 
expi = 1;
load(fullfile(expdirs{expi},datafilestr),'X');
sizeinfo = whos('X');
ngigspersplit = sizeinfo.bytes*nexpspersplit/2^30;
for spliti = 1:nsplits,
  fprintf('Approx amount of space to hold %s split X: %.1f GB\n',splittypes{spliti},ngigspersplit(spliti));
end

% Approx amount of space to hold usertrain split X: 6.9 GB
% Approx amount of space to hold testtrain split X: 3.5 GB
% Approx amount of space to hold test1 split X: 2.4 GB
% Approx amount of space to hold test2 split X: 2.9 GB

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

expi = randsample(nexps,1);
expii = find(reorderinfo.exps==expi);
spliti = exp2splitidx(expi);
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
fprintf('Spot check succeeded!\n');

%% load data

testdata = load('Xtest1.mat');
tmp = load('ytest1.mat');
testdata.y = tmp.y;
featurenames = Xnames;

traindata = load('Xtesttrain.mat');
tmp = load('ytesttrain.mat');
traindata.y = tmp.y;
clear tmp;

%% raw data - no temporal or social information

traindatacollapse = CollapseData_1Animal1Frame(traindata);
testdatacollapse = CollapseData_1Animal1Frame(testdata);

% compute mean and standard deviation of training data
trainmu = nanmean(traindatacollapse.X,2);
trainsig = nanmean(traindatacollapse.X,2);

%% train a logistic regression classifier

maxntrain = 10000;

% markers = {'o','+'};
% colors = [0,0,1;.7,0,0];
% hfig = 123;
% figure(hfig);
% clf;
% nr = round(sqrt(numel(ynames)));
% nc = ceil(numel(ynames)/nr);
% hax = createsubplots(nr,nc,.025);

for yi = 1:numel(ynames),
  
  yname = ynames{yi};
  
  fprintf('%s:\n',yname);
  
  [Xtrain,ytrain] = BasicCleanXy(traindatacollapse.X,traindatacollapse.y(yi,:),trainmu,trainsig);
  labelscurr = unique(ytrain(:));
  nlabelscurr = numel(labelscurr);
  assert(nlabelscurr==2);
      
  % sample some data for training
  [dosample] = BalancedSample(ytrain,maxntrain,labelscurr);
  
%   counts = hist(ytrain,labelscurr);
%   ncurr = numel(ytrain);
%   nsample = min(maxntrain,ncurr);
%   psample = min(1,nsample./counts/nlabelscurr);
%   dosample = rand(ncurr,1);
%   for labeli = 1:nlabelscurr,
%     label = labelscurr(labeli);
%     dosample(ytrain==label) = dosample(ytrain==label) <= psample(labeli);
%   end
%   dosample = dosample ~= 0;
%   nsamplecurr = nnz(dosample);
  Xsample = Xtrain(:,dosample);
  ysample = ytrain(dosample);
  
  [ism,ysampleidx] = ismember(ysample,labelscurr);
  assert(all(ism));
  
  [coeffs,dev,stats] = mnrfit(Xsample',ysampleidx');
  [~,order] = sort(abs(coeffs(2:end)),'descend');
  for ii = 1:3,
    i = order(ii);
    fprintf('%s: coeff = %f, p = %f\n',featurenames{i},coeffs(i+1),stats.p(i+1));
  end
  
  ysampleidxfit = mnrval(coeffs,Xsample');
  [~,ysampleidxfit] = max(ysampleidxfit,[],2);
  ysamplefit = labelscurr(ysampleidxfit)';
  trainconfusionmatrix = ComputeConfusionMatrix(ysample,ysamplefit,labelscurr);
  for labeli = 1:nlabelscurr,
    label = labelscurr(labeli);
    s = sum(trainconfusionmatrix(labeli,:));
    fprintf('Train label = %d, error fraction = %d / %d = %f\n',label,s-trainconfusionmatrix(labeli,labeli),s,(s-trainconfusionmatrix(labeli,labeli))/s);
  end

  [Xtest,ytest] = BasicCleanXy(testdatacollapse.X,testdatacollapse.y(yi,:),trainmu,trainsig);
  ytestidxfit = mnrval(coeffs,Xtest');
  [~,ytestidxfit] = max(ytestidxfit,[],2);
  ytestfit = labelscurr(ytestidxfit)';
  testconfusionmatrix = ComputeConfusionMatrix(ytest,ytestfit,labelscurr);
  for labeli = 1:nlabelscurr,
    label = labelscurr(labeli);
    s = sum(testconfusionmatrix(labeli,:));
    fprintf('Test label = %d, error fraction = %d / %d = %f\n',label,s-testconfusionmatrix(labeli,labeli),s,(s-testconfusionmatrix(labeli,labeli))/s);
  end

%   h = gobjects(nlabelscurr);
%   legs = cell(nlabelscurr);
%   hold(hax(yi),'on');
%   for labeli = 1:nlabelscurr,
%     for predi = 1:nlabelscurr,
%       idxcurr = ytest==labelscurr(labeli)&ytestfit==labelscurr(predi);
%       h(labeli,predi) = plot(hax(yi),Xtest(order(1),idxcurr),Xtest(order(2),idxcurr),...
%         markers{mod(labeli-1,numel(markers))+1},...
%         'Color',colors(mod(predi-1,numel(colors))+1,:),'MarkerSize',4);
%       legs{labeli,predi} = sprintf('True = %d, Pred = %d',labelscurr(labeli),labelscurr(predi));
%     end
%   end
%   if yi == 1,
%     legend(h(:),legs(:));
%   end
%   xlabel(hax(yi),featurenames{order(1)},'Interpreter','none');
%   ylabel(hax(yi),featurenames{order(2)},'Interpreter','none');
%   title(hax(yi),ynames{yi},'Interpreter','none');
%   axisalmosttight([],hax(yi));
%   drawnow;
end


%% some social and temporal information, pca
% 
% maxntrain = 10000;
% winrad = 150;
% ntoffs2 = 10;
% toffs2 = unique(round(logspace(0,log10(winrad),ntoffs2)));
% toffs = [-toffs2(end:-1:1),0,toffs2];
% ntoffs = numel(toffs);
% 
% nfliesclose = 4;
% pcd = 500;
% maxntest = maxntrain;
% 
% for yi = 1:numel(ynames),
%   
%   yname = ynames{yi};
%   fprintf('%s:\n',yname);
%   
%   
%   [dosample,labelscurr] = BalancedSample(traindata.y(yi,:,:),maxntrain);
%   nlabelscurr = numel(labelscurr);
%   dosample(:,:,1:winrad) = false;
%   dosample(:,:,end-winrad+1:end) = false;
%   nsamplescurr = nnz(dosample);
%   ysample = traindata.y(yi,dosample);
%   [ism,ysampleidx] = ismember(ysample,labelscurr);
%   assert(all(ism));
%   
%   [Xsample] = SocialFeatureRepresentation(traindata.X,Xnames,'toffs',toffs,'dosample',dosample,'nfliesclose',nfliesclose,'verbose',0);
%   nfeaturesperframe = size(Xsample,1);
%   nfeaturestotal = nfeaturesperframe*nfliesclose*ntoffs;
%   
%   % z-score
%   Zsample = reshape(Xsample,[nfeaturestotal,nsamplescurr])';
%   mu = nanmean(Zsample,1);
%   sig = nanstd(Zsample,1,1);
%   sig(sig<eps) = 1;
%   Zsample = (Zsample-mu)./sig;
%   % fill missing data
%   Zsample(isnan(Zsample)) = 0;
%   
%   % pca
%   %pcd = min(nfeaturestotal,nsamplescurr);
%   assert(pcd <= nfeaturestotal);
%   fprintf('PCA...\n');
%   [U,lambda] = pca(Zsample,'centered',false,'Economy',true,'NumComponents',pcd,'Rows','all');
%   Xpca = Zsample*U;
%   
%   fprintf('Logistic regression...\n');
%   [coeffs,dev,stats] = mnrfit(Xpca,ysampleidx');
%   
%   [~,order] = sort(abs(coeffs(2:end)),'descend');
%   for ii = 1:3,
%     i = order(ii);
%     fprintf('PC %d: coeff = %f, p = %f\n',i,coeffs(i+1),stats.p(i+1));
%   end
%   
%   ysampleidxfit = mnrval(coeffs,Xpca);
%   [~,ysampleidxfit] = max(ysampleidxfit,[],2);
%   ysamplefit = labelscurr(ysampleidxfit)';
%   trainconfusionmatrix = ComputeConfusionMatrix(ysample,ysamplefit,labelscurr);
%   for labeli = 1:nlabelscurr,
%     label = labelscurr(labeli);
%     s = sum(trainconfusionmatrix(labeli,:));
%     fprintf('Train label = %d, error fraction = %d / %d = %f\n',label,s-trainconfusionmatrix(labeli,labeli),s,(s-trainconfusionmatrix(labeli,labeli))/s);
%   end
%   
%   [dosampletest] = BalancedSample(testdata.y(yi,:,:),maxntest,labelscurr);
%   dosampletest(:,:,1:winrad) = false;
%   dosampletest(:,:,end-winrad+1:end) = false;
%   nsamplestest = nnz(dosampletest);
%   
%   [Xtest] = SocialFeatureRepresentation(testdata.X,Xnames,'toffs',toffs,'dosample',dosampletest,'nfliesclose',nfliesclose,'verbose',0);
%   % z-score
%   Ztest = reshape(Xtest,[nfeaturestotal,nsamplestest])';
%   Ztest = (Ztest-mu)./sig;
%   % fill missing data
%   Ztest(isnan(Ztest)) = 0;
%   % PCA
%   Xtestpca = Ztest*U;
%   
%   ytest = testdata.y(yi,dosampletest);
%   ytestidxfit = mnrval(coeffs,Xtestpca);
%   [~,ytestidxfit] = max(ytestidxfit,[],2);
%   ytestfit = labelscurr(ytestidxfit)';
%   
%   testconfusionmatrix = ComputeConfusionMatrix(ytest,ytestfit,labelscurr);
%   for labeli = 1:nlabelscurr,
%     label = labelscurr(labeli);
%     s = sum(testconfusionmatrix(labeli,:));
%     fprintf('Test label = %d, error fraction = %d / %d = %f\n',label,s-testconfusionmatrix(labeli,labeli),s,(s-testconfusionmatrix(labeli,labeli))/s);
%   end
%   
% end

%% some social and temporal information, pca, on cluster

maxntrain = 10000;
winrad = 150;
% winradpool = 1000;
ntoffs2 = 10;
% ntoffspool2 = 20;
nfliesclose = 4;
pcd = 500;
maxntest = maxntrain;
matlabpath = '/misc/local/matlab-2019a/bin/matlab';
cwd = pwd;

usecluster = false;

testname = sprintf('winrad%03d',winrad);

if usecluster,
  loadfile = fullfile(cwd,'SocialPCATestData.mat');
  save(loadfile,'maxntrain','winrad','ntoffs2','nfliesclose','pcd','maxntest','Xnames','ynames','traindata','testdata');%,'winradpool','ntoffspool2');
else
  data = struct;
  data.maxntrain = maxntrain;
  data.winrad = winrad;
  data.ntoffs2 = ntoffs2;
  data.nfliesclose = nfliesclose;
  data.pcd = pcd;
  data.maxntest = maxntest;
  data.Xnames = Xnames;
  data.ynames = ynames;
  data.traindata = traindata;
  data.testdata = testdata;
end

for yi = 1:numel(ynames)-1,
  
  savefile = fullfile(cwd,sprintf('SocialPCATestResult%02d_%s.mat',yi,testname));
  if usecluster,
    logfile = fullfile(cwd,'run',sprintf('SocialPCATestResult%02d_%s.out',yi,testname));
    resfile = fullfile(cwd,'run',sprintf('SocialPCATestResult%02d_%s_bsub.out',yi,testname));
    reserrfile = fullfile(cwd,'run',sprintf('SocialPCATestResult%02d_%s_bsub.err',yi,testname));  
    matlabcmd = sprintf('SetUpPaths; SocialPCATest(%d,''%s'',''%s'');',yi,loadfile,savefile);
    jobcmd = sprintf('cd %s; %s -nodisplay -batch "%s exit;" > %s 2>&1',cwd,matlabpath,matlabcmd,logfile);
    bsubcmd = sprintf('bsub -n 4 -J socialtest%02d_%s -o %s -e %s "%s"',yi,testname,resfile,reserrfile,strrep(jobcmd,'"','\"'));
    sshcmd = sprintf('ssh login1 "%s"',strrep(strrep(bsubcmd,'\','\\'),'"','\"'));
    disp(sshcmd);
    unix(sshcmd);
  else
    SocialPCATest(yi,data,savefile);
  end
  
end

for yi = 1:numel(ynames),
  
  savefile = fullfile(cwd,sprintf('SocialPCATestResult%02d.mat',yi));
  fprintf('%s:\n',ynames{yi});
  load(savefile);
  [~,order] = sort(abs(coeffs(2:end)),'descend');
  for ii = 1:3,
    i = order(ii);
    fprintf('PC %d: coeff = %f, p = %f\n',i,coeffs(i+1),stats.p(i+1));
  end
  for labeli = 1:nlabelscurr,
    label = labelscurr(labeli);
    s = sum(trainconfusionmatrix(labeli,:));
    fprintf('Train label = %d, error fraction = %d / %d = %f\n',label,s-trainconfusionmatrix(labeli,labeli),s,(s-trainconfusionmatrix(labeli,labeli))/s);
  end
  for labeli = 1:nlabelscurr,
    label = labelscurr(labeli);
    s = sum(testconfusionmatrix(labeli,:));
    fprintf('Test label = %d, error fraction = %d / %d = %f\n',label,s-testconfusionmatrix(labeli,labeli),s,(s-testconfusionmatrix(labeli,labeli))/s);
  end
  
end