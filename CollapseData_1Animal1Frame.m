function data = CollapseData_1Animal1Frame(data)

[dX,nflies,nframes] = size(data.X);
dy = size(data.y,1);
data.X = reshape(data.X,[dX,nflies*nframes]);
data.y = reshape(data.y,[dy,nflies*nframes]);
data.ids = reshape(data.ids,[1,nflies*nframes]);
nodata = isnan(data.ids);
data.X(:,nodata) = [];
data.y(:,nodata) = [];
data.ids(nodata) = [];
frames = repmat(data.frames,[nflies,1]);
frames(nodata) = [];
data.frames = frames;
videoidx = repmat(data.videoidx,[nflies,1]);
videoidx(nodata) = [];
data.videoidx = videoidx;