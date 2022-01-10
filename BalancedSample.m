% sample some data for training, with equal sample sizes per label
function [dosample,labels] = BalancedSample(y,nsample,labels)

isdata = ~isnan(y);
if nargin < 3,
  labels = unique(y(isdata));
end
nlabels = numel(labels);
counts = hist(y(isdata),labels);
n = nnz(isdata);
nsample1 = min(nsample,n);
psample = min(1,nsample1./counts/nlabels);
dosample1 = rand(n,1);
for labeli = 1:nlabels,
  label = labels(labeli);
  dosample1(y(isdata)==label) = dosample1(y(isdata)==label) <= psample(labeli);
end
dosample1 = dosample1 ~= 0;
dosample = false(size(y));
dosample(isdata) = dosample1;
