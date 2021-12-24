function [bestassignment,finalcost,finalstats] = AssignSplits(nboutsperexp,desiredexpfracs,desiredlabelfracs,varargin)

[costnolabels,costnolabelvals,costnoexps,weightexp,weightlabel,weightlabelval,...
  maxniters,pflipextra,pacceptinit,pacceptfinal] = ...
  myparse(varargin,'costnolabels',1000,...
  'costnolabelvals',200,...
  'costnoexps',10000,...
  'weightexp',1000,...
  'weightlabel',10,...
  'weightlabelval',5,...
  'maxniters',100000,...
  'pflipextra',.1,...
  'pacceptinit',.5,...
  'pacceptfinal',.01);

desiredexpfracs = desiredexpfracs(:);
desiredlabelfracs = desiredlabelfracs(:);
splitneedslabels = desiredlabelfracs > 0;
nsplits = numel(desiredexpfracs);
[nexps,nlabels,nvals] = size(nboutsperexp);
Zboutsperlabelval = max(1,sum(nboutsperexp,1));
Zboutsperlabel = max(1,sum(sum(nboutsperexp,1),3));

assert(nexps>=nsplits);
assignment = randsample(nsplits,nexps,true,desiredexpfracs);
cost = ComputeCost(assignment);
bestassignment = assignment;
bestcost = cost;
pacceptperiter = fliplr(logspace(log10(pacceptfinal),log10(pacceptinit),maxniters));
for iter = 1:maxniters,
  paccept = pacceptperiter(iter);
  costprev = cost;
  newassignment = MutateAssignment(assignment);
  cost = ComputeCost(newassignment);
  if cost < costprev,
    assignment = newassignment;
    if cost < bestcost,
      bestcost = cost;
      bestassignment = assignment;
      fprintf('Iter %d, best cost = %f\n',iter,bestcost);
    end
  else
    if rand(1) < paccept,
      assignment = newassignment;
    end
  end
end

[finalcost,finalstats] = ComputeCost(bestassignment);

  function y = MutateAssignment(x)
    
    y = x;
    i = randsample(nexps,1);
    doswitch = false(1,nexps);
    doswitch(i) = true;
    doswitch([1:i-1,i+1:nexps]) = rand(1,nexps-1)<=pflipextra;
    for i = find(doswitch),
      s = randsample(nsplits-1,1);
      if s >= x(i),
        s = s+1;
      end
      y(i) = s;
    end

  end

  function [cost,varargout] = ComputeCost(x)
    
    nexpspersplit = zeros(nsplits,1);
    nboutspersplit = zeros(nsplits,nlabels,nvals);
    for spliti = 1:nsplits,
      %[nexps,nlabels,nvals] = size(nboutsperexp);
      nboutspersplit(spliti,:,:) = sum(nboutsperexp(x==spliti,:,:),1);
      nexpspersplit(spliti) = nnz(x==spliti);
    end
    cost = 0;
    nsplitsnoexp = sum(nexpspersplit==0);
    cost = cost + nsplitsnoexp*costnoexps;
    nsplitsnolabel = nnz(sum(nboutspersplit(splitneedslabels,:,:),3)==0);
    cost = cost + nsplitsnolabel/nlabels*costnolabels;
    nsplitsnolabelval = nnz(nboutspersplit==0);
    cost = cost + nsplitsnolabelval/nlabels/nvals*costnolabelvals;
    splitexpfracs = nexpspersplit/nexps;
    cost = cost + sum(abs( splitexpfracs - desiredexpfracs ))*weightexp;
    splitlabelfracs = sum(nboutspersplit,3)./Zboutsperlabel;
    cost = cost + sum(sum(abs( splitlabelfracs - desiredlabelfracs )))/nlabels*weightlabel;
    splitlabelvalfracs = nboutspersplit./Zboutsperlabelval;
    cost = cost + sum(sum(sum(abs( splitlabelvalfracs - desiredlabelfracs ))))/nlabels/nvals*weightlabelval;
    
    if nargout > 1,
      stats = struct;
      stats.nsplitsnoexp = nsplitsnoexp;
      stats.nsplitsnolabels = nsplitsnolabel;
      stats.nsplitsnolabelval = nsplitsnolabelval;
      stats.splitexpfracs = splitexpfracs;
      stats.splitlabelfracs = splitlabelfracs;
      stats.splitlabelvalfracs = splitlabelvalfracs;
      varargout{1} = stats;
    end
    
    
  end

end