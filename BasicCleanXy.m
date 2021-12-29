function [X,y,nolabel] = BasicCleanXy(X,y,trainmu,trainsig)

nolabel = isnan(y);
y(nolabel) = [];
X = X(:,~nolabel);
X = (X - trainmu)./trainsig;
X(isnan(X)) = 0;