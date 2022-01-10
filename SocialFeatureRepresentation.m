function [X,dosample] = SocialFeatureRepresentation(X0,Xnames,varargin)

[winrad,dosample,nfliesclose,verbose,toffs,idxsample,toffspool] = ...
  myparse(varargin,'winrad',10,'dosample',[],'nfliesclose',[],...
  'verbose',1,'toffs',[],'idxsample',[],'toffspool',0);

[nfeatures0,nflies,nframes] = size(X0);
if isempty(dosample) && isempty(idxsample),
  dosample = false(nflies,nframes);
  dosample(:,winrad+1:nframes-winrad) = true;
end
if isempty(idxsample),
  idxsample = find(dosample);
end
if isempty(toffs),
  toffs = -winrad:winrad;
end
ntoffs = numel(toffs);
% must include current frame
tidx0 = find(toffs==0);
assert(nnz(tidx0)==1);
ntoffspool = numel(toffspool);
nsamples = numel(idxsample);

%assert(~any(any(any(isnan(X0),1) & ~all(isnan(X0),1))));
isdata = ~all(isnan(X0),1);
if isempty(nfliesclose),
  nfliesclose = min(sum(isdata,2),[],3);
end

% for finding the center and orientation of an animal
bodynames = {
  'left_shoulder'
  'right_shoulder'
  'end_notum'
  'end_abdomen'
};

bodynames_x = cellfun(@(x) [x,'_x_mm'],bodynames,'Uni',0);
bodynames_y = cellfun(@(x) [x,'_y_mm'],bodynames,'Uni',0);
[ism,bodyidx_x] = ismember(bodynames_x,Xnames);
assert(all(ism));
[ism,bodyidx_y] = ismember(bodynames_y,Xnames);
assert(all(ism));

centerx = mean(X0(bodyidx_x,idxsample),1);
centery = mean(X0(bodyidx_y,idxsample),1);
assert(~any(isnan(centerx(isdata(idxsample)))));
assert(~any(isnan(centery(isdata(idxsample)))));

dx = X0(bodyidx_x,idxsample)-centerx;
dy = X0(bodyidx_y,idxsample)-centery;
dx2 = sum(dx.^2,1);
dxy = sum(dx.*dy,1);
dy2 = sum(dy.^2,1);
[~,v] = eigs_2x2(cat(1,dx2,dxy,dxy,dy2));
v = reshape(v(:,1,:),[2,nsamples]);
theta = atan2(v(2,:),v(1,:));
theta1 = atan2(getX(X0(:,idxsample),Xnames,'antennae_y_mm')-getX(X0(:,idxsample),Xnames,'end_abdomen_y_mm'),...
  getX(X0(:,idxsample),Xnames,'antennae_x_mm')-getX(X0(:,idxsample),Xnames,'end_abdomen_x_mm'));
addtheta = abs(modrange(theta+pi-theta1,-pi,pi)) < abs(modrange(theta-theta1,-pi,pi));
theta(addtheta) = theta(addtheta)+pi;
theta = modrange(theta,-pi,pi);
costheta = cos(theta);
sintheta = sin(theta);

idxx = ~cellfun(@isempty,regexp(Xnames,'_x(_mm)?$','once'));
idxy = ~cellfun(@isempty,regexp(Xnames,'_y(_mm)?$','once'));
nlandmarks = nnz(idxx);


legtipnames = regexp(Xnames,'^(tip_.*)_x_mm$','tokens','once');
legtipnames(cellfun(@isempty,legtipnames))=[];
legtipnames = [legtipnames{:}];
legtipnames_x = cellfun(@(x) [x,'_x_mm'],legtipnames,'Uni',0);
legtipnames_y = cellfun(@(x) [x,'_y_mm'],legtipnames,'Uni',0);
[ism,legtipidx_x] = ismember(legtipnames_x,Xnames(idxx));
assert(all(ism));
[ism,legtipidx_y] = ismember(legtipnames_y,Xnames(idxy));
assert(all(ism));
nlegtips = numel(legtipnames);

keyptnames = {'antennae','end_abdomen'};
keyptnames_x = cellfun(@(x) [x,'_x_mm'],keyptnames,'Uni',0);
keyptnames_y = cellfun(@(x) [x,'_y_mm'],keyptnames,'Uni',0);
[ism,keyptidx_x] = ismember(keyptnames_x,Xnames(idxx));
assert(all(ism));
[ism,keyptidx_y] = ismember(keyptnames_y,Xnames(idxy));
assert(all(ism));
nkeypts = numel(keyptnames);

% xraw is nfeatures0, xcentered, ycentered are nlandmarks, distlmk is
% nlandmarks^2
% remove nlandmarks for dists that will always be 0
nfeaturesperflyframe = nfeatures0+2*nlandmarks+nkeypts*nlegtips + nkeypts^2 + nkeypts;
X = nan([nfeaturesperflyframe,nfliesclose,ntoffs,ntoffspool,nsamples]);

if verbose,
  vt = tic;
end

for i = 1:nsamples,
  if verbose && toc(vt) > 1,
    vt = tic;
    fprintf('Sample %d / %d\n',i,nsamples);
  end
  [fly,f0] = ind2sub([nflies,nframes],idxsample(i));
  
  for ipool = 1:ntoffspool,
    toffpool = toffspool(ipool);
    f = f0+toffpool;
    
    % skip out of bounds data
    if f+min(toffs) < 1 || f+max(toffs) > nframes,
      continue;
    end

    xraw = X0(:,:,f+toffs);
    dx = xraw(idxx,:,:)-centerx(i);
    dy = xraw(idxy,:,:)-centery(i);
    xcentered = costheta(i).*dx+sintheta(i).*dy;
    ycentered = -sintheta(i).*dx + costheta(i).*dy;
    %     vycentered = diff(xcentered,1,3);
    %     vycentered = diff(xcentered,1,3);
    
    % all pairs distances between landmarks, current frame
    % nlandmarks x nlandmarks x nflies
    distlmkcurr = sqrt((permute(xcentered(:,:,tidx0),[4,1,2,3])-permute(xcentered(:,fly,tidx0),[1,2,4,3])).^2 + ...
      (permute(ycentered(:,:,tidx0),[4,1,2,3])-permute(ycentered(:,fly,tidx0),[1,2,4,3])).^2);
    
    % all frames, distance between this fly's leg tips to all flies' keypts
    % nlegtips x nkeypts x nflies x ntoffs
    distlegtip2keypt = sqrt((permute(xcentered(keyptidx_x,:,:),[4,1,2,3])-permute(xcentered(legtipidx_x,fly,:),[1,2,4,3])).^2 + ...
      (permute(ycentered(keyptidx_y,:,:),[4,1,2,3])-permute(ycentered(legtipidx_y,fly,:),[1,2,4,3])).^2);
    
    % all frames, distance between this fly's keypts to keypts
    % nkeypts x nkeypts x nflies x ntoffs
    distkeypt2keypt = sqrt((permute(xcentered(keyptidx_x,:,:),[4,1,2,3])-permute(xcentered(keyptidx_x,fly,:),[1,2,4,3])).^2 + ...
      (permute(ycentered(keyptidx_y,:,:),[4,1,2,3])-permute(ycentered(keyptidx_y,fly,:),[1,2,4,3])).^2);
    
    % all frames, distance between this fly's key points to closest leg tips
    % nkeypts x 1 x nflies x ntoffs
    distkeypt2legtip = min(sqrt((permute(xcentered(legtipidx_x,:,:),[4,1,2,3])-permute(xcentered(keyptidx_x,fly,:),[1,2,4,3])).^2 + ...
      (permute(ycentered(legtipidx_y,:,:),[4,1,2,3])-permute(ycentered(keyptidx_y,fly,:),[1,2,4,3])).^2),[],2);
    
    distlmk = cat(1,reshape(distlegtip2keypt,[nlegtips*nkeypts,nflies,ntoffs]),...
      reshape(distkeypt2keypt,[nkeypts*nkeypts,nflies,ntoffs]),...
      reshape(distkeypt2legtip,[nkeypts,nflies,ntoffs]));
        
    % sort based on min landmark to landmark distance on center frame
    distsortlmk = min(min(distlmkcurr(:,:,[1:fly-1,fly+1:end]),[],1),[],2);
    [mindist,order] = sort(distsortlmk);
    order(order>=fly) = order(order>=fly)+1;
    order = [fly;order(:)];
    order = order(1:nfliesclose);
    
    xraw = xraw(:,order,:);
    xcentered = xcentered(:,order,:);
    ycentered = ycentered(:,order,:);
    %distlmk = distlmk(:,:,order,:);
    %distlmk = reshape(distlmk,[nlandmarks^2,nfliesclose,ntoffs]);
    distlmk = distlmk(:,order,:);
    
    Xcurr = cat(1,xraw,xcentered,ycentered,distlmk);
    X(:,:,:,ipool,i) = Xcurr;
  end
end
  
function VisualizeFeatures(fly,Xnames,varargin),

skeledges = [
  6     7
  9    13
  10    11
  11    16
  6    10
  8     9
  6     8
  4     6
  1     2
  1     6
  4    17
  5    12
  6    15
  6    14
  1     3
  5     6
  ];
nskeledges = size(skeledges,1);

[xraw,xcentered,ycentered,nskip] = myparse(varargin,'xraw',[],...
  'xcentered',[],'ycentered',[],'nskip',5);

if ~isempty(xraw),
  [~,nflies,wframes] = size(xraw);
elseif ~isempty(xcentered),
  [~,nflies,wframes] = size(xcentered);
else
  error('No input given');
end
winrad = (wframes-1)/2;
 
nax = double(~isempty(xraw)) + double(~isempty(xcentered));
clf;
hax = createsubplots(nax,1,.02);
axi = 1;
flycolors1 = jet(64);
flycolors1 = flycolors1(round(linspace(1,64,nflies)),:);
flycolors = zeros(nflies,3);
flycolors(fly,:) = flycolors1(end,:);
flycolors([1:fly-1,fly+1:end],:) = flycolors1(1:end-1,:);
frameweights = linspace(.5,1,winrad+1);
frameweights = [frameweights,frameweights(end-1:-1:1)];
js = 1:nskip:2*winrad+1;
js(js==winrad+1) = [];
js(end+1) = winrad+1;

if ~isempty(xraw),
  for j = js,
    for flyi = 1:nflies,
      color = flycolors(flyi,:)*frameweights(j) + (1-frameweights(j));
      h = drawflyo(getX(xraw(:,flyi,j),Xnames,'x_mm'),getX(xraw(:,flyi,j),Xnames,'y_mm'),...
        atan2(getX(xraw(:,flyi,j),Xnames,'sin_ori'),getX(xraw(:,flyi,j),Xnames,'cos_ori')),...
        getX(xraw(:,flyi,j),Xnames,'maj_ax_mm')/4,getX(xraw(:,flyi,j),Xnames,'min_ax_mm')/4,...
        'Color',color,'parent',hax(axi));
      if fly == flyi && j == wframes,
        set(h,'LineWidth',2);
      end
      hold(hax(axi),'on');
    end
  end
  axis(hax(axi),'equal');
  box(hax(axi),'off');
  set(hax(axi),'Color','k');
  axi = axi + 1;
end

if ~isempty(xcentered),
  for j = js,
    for flyi = 1:nflies,
      color = flycolors(flyi,:)*frameweights(j) + (1-frameweights(j));
      h = plot(hax(axi),[reshape(xcentered(skeledges,flyi,j),[nskeledges,2]),nan(nskeledges,1)]',...
        [reshape(ycentered(skeledges,flyi,j),[nskeledges,2]),nan(nskeledges,1)]',...
        '-','Color',color);
      if fly == flyi && j == wframes,
        set(h,'LineWidth',2);
      end
      hold(hax(axi),'on');
    end
  end
  axis(hax(axi),'equal');
  set(hax(axi),'Color','k');
  box(hax(axi),'off');  
end

if nax > 1,
  linkaxes(hax);
end