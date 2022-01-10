function SocialPCATest(yi,loadfile,savefile)

% maxntrain
% maxntest
% winrad
% ntoffs2
% nfliesclose
% pcd
% Xnames
% ynames
% traindata
% testdata
% winradpool

if ischar(loadfile),
  load(loadfile);
else
  fns = fieldnames(loadfile);
  for i = 1:numel(fns),
    fn = fns{i};
    eval(sprintf('%s = loadfile.(fn);',fn));
  end
end
rng('default');

% if ~exist('ntoffspool2','var'),
%   ntoffspool2 = 0;
% end
% if ~exist('fracpooltrain','var'),
%   fracpooltrain = 1;
% end

% train classifier on short snippets
if winrad < 5*ntoffs2,
  toffs2 = unique(round(linspace(1,winrad,ntoffs2)));
else
  toffs2 = unique(round(logspace(0,log10(winrad),ntoffs2)));
end
toffs = [-toffs2(end:-1:1),0,toffs2];

% % pool results over many windows
% toffspool2 = unique(round(linspace(1,winradpool,ntoffspool2)));
% toffspool = [-toffspool2(end:-1:1),0,toffspool2];

ntoffs = numel(toffs);
% ntoffspool = numel(toffspool);
  
yname = ynames{yi};
fprintf('%s:\n',yname);
 
[dosample,labelscurr] = BalancedSample(traindata.y(yi,:,:),maxntrain);
nlabelscurr = numel(labelscurr);
dosample(:,:,1:winrad) = false;
dosample(:,:,end-winrad+1:end) = false;
nsamplescurr = nnz(dosample);
ysample = traindata.y(yi,dosample);
[ism,ysampleidx] = ismember(ysample,labelscurr);
assert(all(ism));
  
[Xsample] = SocialFeatureRepresentation(traindata.X,Xnames,'toffs',toffs,'dosample',dosample,'nfliesclose',nfliesclose,'verbose',0);%,'toffspool',toffspool);
nfeaturesperframe = size(Xsample,1);
nfeaturestotal = nfeaturesperframe*nfliesclose*ntoffs;
  
% z-score
Zsample = reshape(Xsample,[nfeaturestotal,nsamplescurr])';
%Zsample = reshape(Xsample,[nfeaturestotal,ntoffspool*nsamplescurr])';
mu = nanmean(Zsample,1);
sig = nanstd(Zsample,1,1);
sig(sig<eps) = 1;
Zsample = (Zsample-mu)./sig;
% fill missing data
Zsample(isnan(Zsample)) = 0;

% % subsample for training
% if fracpooltrain < 1,
%   Zsample = reshape(Zsample,[ntoffspool,nsamplescurr,nfeaturestotal]);
%   ntrainpoolsample = max(1,round(fracpooltrain*ntoffspool));
%   dosamplepool = false(ntoffspool,nsamplescurr);
%   Zsampletrain = nan([ntrainpoolsample,nsamplescurr,nfeaturestotal]);
%   for i = 1:nsamplescurr,
%     dosamplepool(randperm(nsamplescurr,ntrainpoolsample),i) = true;
%     Zsampletrain(:,i,:) = Zsample(dosamplepool(:,i),i,:);
%   end
%   Zsample = reshape(Zsample,[ntoffspool*nsamplescurr,nfeaturestotal]);
%   Zsampletrain = reshape(Zsampletrain,[ntrainpoolsample*nsamplescurr,nfeaturestotal]);
% else
%   ntrainpoolsample = ntoffspool;
%   Zsampletrain = Zsample;
%   dosamplepool = true(ntoffspool,nsamplescurr);
% end

% pca
%pcd = min(nfeaturestotal,nsamplescurr);
assert(pcd <= nfeaturestotal);
fprintf('PCA...\n');
% [U,lambda] = pca(Zsampletrain,'centered',false,'Economy',true,'NumComponents',pcd,'Rows','all');
% Xpcatrain = Zsampletrain*U;
% Xpca = Zsample*U;
[U,lambda] = pca(Zsample,'centered',false,'Economy',true,'NumComponents',pcd,'Rows','all');
Xpca = Zsample*U;

fprintf('Logistic regression...\n');
% [coeffs,dev,stats] = mnrfit(Xpcatrain,ysampleidx');
[coeffs,dev,stats] = mnrfit(Xpca,ysampleidx');

[~,order] = sort(abs(coeffs(2:end)),'descend');
for ii = 1:3,
  i = order(ii);
  fprintf('PC %d: coeff = %f, p = %f\n',i,coeffs(i+1),stats.p(i+1));
end
  
ysampleidxfit = mnrval(coeffs,Xpca);
% ysampleidxfit = reshape(ysampleidxfit,[ntoffspool,nsamplescurr,nlabelscurr]);
% ysampleidxfit = permute(sum(ysampleidxfit,1),[2,3,1]);
[~,ysampleidxfit] = max(ysampleidxfit,[],2);
ysamplefit = labelscurr(ysampleidxfit)';
trainconfusionmatrix = ComputeConfusionMatrix(ysample,ysamplefit,labelscurr);
for labeli = 1:nlabelscurr,
  label = labelscurr(labeli);
  s = sum(trainconfusionmatrix(labeli,:));
  fprintf('Train label = %d, error fraction = %d / %d = %f\n',label,s-trainconfusionmatrix(labeli,labeli),s,(s-trainconfusionmatrix(labeli,labeli))/s);
end

[dosampletest] = BalancedSample(testdata.y(yi,:,:),maxntest,labelscurr);
dosampletest(:,:,1:winrad) = false;
dosampletest(:,:,end-winrad+1:end) = false;
nsamplestest = nnz(dosampletest);

[Xtest] = SocialFeatureRepresentation(testdata.X,Xnames,'toffs',toffs,'dosample',dosampletest,'nfliesclose',nfliesclose,'verbose',0);%,'toffspool',toffspool);
% z-score
Ztest = reshape(Xtest,[nfeaturestotal,nsamplestest])';
%Ztest = reshape(Xtest,[nfeaturestotal,ntoffspool*nsamplestest])';
Ztest = (Ztest-mu)./sig;
% fill missing data
Ztest(isnan(Ztest)) = 0;
% PCA
Xtestpca = Ztest*U;

ytest = testdata.y(yi,dosampletest);
ytestidxfit = mnrval(coeffs,Xtestpca);
%ytestidxfit = reshape(ytestidxfit,[ntoffspool,nsamplestest,nlabelscurr]);
%ytestidxfit = permute(sum(ytestidxfit,1),[2,3,1]);

[~,ytestidxfit] = max(ytestidxfit,[],2);
ytestfit = labelscurr(ytestidxfit)';

testconfusionmatrix = ComputeConfusionMatrix(ytest,ytestfit,labelscurr);
for labeli = 1:nlabelscurr,
  label = labelscurr(labeli);
  s = sum(testconfusionmatrix(labeli,:));
  fprintf('Test label = %d, error fraction = %d / %d = %f\n',label,s-testconfusionmatrix(labeli,labeli),s,(s-testconfusionmatrix(labeli,labeli))/s);
end

save(savefile,'trainconfusionmatrix','testconfusionmatrix','U','coeffs');