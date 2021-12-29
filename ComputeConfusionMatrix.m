function confusionmatrix = ComputeConfusionMatrix(ytrue,ypred,labels)
nlabels = numel(labels);
confusionmatrix = nan(nlabels);
for truelabeli = 1:nlabels,
  truelabel = labels(truelabeli);
  for predlabeli = 1:nlabels,
    predlabel = labels(predlabeli);
    confusionmatrix(truelabeli,predlabeli) = nnz(ypred==predlabel & ytrue == truelabel);
  end
end