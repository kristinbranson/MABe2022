function Xsub = getX(X,Xnames,name)

if ischar(name),
  name = {name};
end
[ism,idx] = ismember(name,Xnames);
assert(all(ism));
Xsub = X(idx,:,:);