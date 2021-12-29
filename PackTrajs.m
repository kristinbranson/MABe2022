function [assignment,bestcost] = PackTrajs(sfs,efs,pos)

ntrajs = numel(sfs);
nframes = max(efs);
iscomplete = sfs==1 & efs == nframes;
if any(iscomplete),
  idxcomplete = find(iscomplete);
  idxtracklet = find(~iscomplete);
  ncomplete = numel(idxcomplete);
  ntracklet = numel(idxtracklet);
  if ntracklet == 0,
    assignment = 1:ncomplete;
    bestcost = 0;
    fprintf('No partial tracklets.\n');
    return;
  end
  assignment = nan(1,ntrajs);
  assignment(idxcomplete) = 1:ncomplete;
  [assignmentrest,bestcost] = PackTrajs(sfs(idxtracklet),efs(idxtracklet),pos(:,idxtracklet,:));
  assignment(idxtracklet) = assignmentrest+ncomplete;
  fprintf('Final cost %f for complete assignment %s\n',bestcost,mat2str(assignment));

  return;
end

isdata = zeros(ntrajs,nframes);
for i = 1:ntrajs,
  isdata(i,sfs(i):efs(i)) = true;
end
minnflies = max(sum(isdata,1));

nflies = minnflies;
for nflies = minnflies:ntrajs,
  isused = false(nflies,nframes);
  cost = zeros(nflies,1);
  [bestcost,assignment] = PackTrajsSearch(nan(1,ntrajs),isused,cost,nflies,sfs,efs,pos);
  fprintf('N. flies = %d, cost %f for assignment %s\n',nflies,bestcost,mat2str(assignment));
  if ~isinf(bestcost),
    break;
  end
end

function [bestcost,bestassignment] = PackTrajsSearch(assignment,isused,cost,nflies,sfs,efs,pos)

% persistent idxsearched
% if exist('firstrun','var') && firstrun,
%   idxsearched = [];
% end
% idxv = assignment;
% idxv(isnan(idxv)) = 0;
% idxcurr = sub2indv(repmat(nflies+1,[1,numel(assignment)]),idxv+1);
% assert(~ismember(idxcurr,idxsearched));
% idxsearched(end+1) = idxcurr;

%fprintf('Base assignment = %s, cost = %s\n',mat2str(assignment),mat2str(cost));

bestassignment = assignment;
isdone = all(~isnan(assignment));
if isdone,
  fprintf('Complete assignment = %s, cost = sum(%s) = %f\n',mat2str(assignment),mat2str(cost),sum(cost));
  bestcost = sum(cost);
  return;
end

bestcost = inf;
traji = find(isnan(assignment),1);
isunassigned = false(1,nflies);
for fly = 1:nflies,
  isunassigned(fly) = ~any(assignment==fly);
end

triedunassigned = false;
for fly = 1:nflies,
  if any(isused(fly,sfs(traji):efs(traji)),2),
    continue;
  end
  if isunassigned(fly),
    if triedunassigned,
      continue;
    end
    triedunassigned = true;
  end
  childcost = cost;
  childcost(fly) = 0;
  
  trajidx = [traji,find(assignment==fly)];
  if numel(trajidx) >= 2,
    
    [~,order] = sort(sfs(trajidx));
    for j = 1:numel(trajidx)-1,
      trajj1 = trajidx(order(j));
      trajj2 = trajidx(order(j+1));
      dist = sqrt(sum((pos(:,trajj1,efs(trajj1))-pos(:,trajj2,sfs(trajj2))).^2,1));
      childcost(fly) = childcost(fly) + dist;
    end
  end
  
  child = assignment;
  child(traji) = fly;
  childisused = isused;
  childisused(fly,sfs(traji):efs(traji)) = true;
  [childcost,child] = PackTrajsSearch(child,childisused,childcost,nflies,sfs,efs,pos);
  if sum(childcost) < bestcost,
    bestcost = sum(childcost);
    bestassignment = child;
  end
  
end
