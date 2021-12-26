%% file paths

datafile = 'SocialFlyBubbleExperiments_v2.csv';
analysis_protocol = 'current_non_olympiad_dickson_VNC';
settingsdir = '/groups/branson/home/bransonk/behavioranalysis/code/FlyDiscoAnalysis/settings';
rootdatadir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data';
finaldatadir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/sharedata20211226';
datalocparamsfilestr = 'dataloc_params.txt';
fdapath = '/groups/branson/home/bransonk/behavioranalysis/code/FlyDiscoAnalysis'; 
aptpath = '/groups/branson/home/bransonk/tracking/code/APT';
rundir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/run';
matlabpath = '/misc/local/matlab-2019a/bin/matlab';
addpath(aptpath);
APT.setpath;
addpath(fdapath);
modpath;
cwd = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022';

reg_settingsdir = '/groups/branson/home/robiea/Code_versioned/FlyDiscoAnalysis/settings';
reg_trp_analysis_protocol = '20211014_flybubbleRed_noChr';
reg_chr_analysis_protocol_red = '20211014_flybubbleRed_LED';
reg_chr_analysis_protocol_rgb = '20210806_flybubble_LED';

lblfile = '/groups/branson/home/robiea/Projects_data/Labeler_APT/Nan_labelprojects_touchinglabels/20210916/multitarget_bubble_training_20210523_allGT_AR_params20210920.lbl';
apttrackerdir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/apttracker';

bindpaths = {'/nrs/branson','/groups/branson'};
singularityimg = '/groups/branson/bransonlab/apt/sif/prod.sif';

isdoneprotocols = {'20210806_flybubble_LED_NorpA','20210806_flybubble_LED'};

labeljabfiles = {
  '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/aggression2.jab'
  '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/chase.jab'
  '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/courtship_v2pt3.jab'
  '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/highfence2.6.jab'
  '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/wingextension.jab'
  '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/wingflick_v1pt9.jab'
  };

%% parameters

domakecopies = false;
docopyapttracker = false;

%% run scripts

if ~exist(rundir,'dir'),
  mkdir(rundir);
end

timestamp = datestr(now,'yyyymmddTHHMMSS');

%% read in info about experiments to analyze

fid = fopen(datafile,'r');
s = fgetl(fid);
headers = regexp(s,',','split');
expinfo = [];
expdiri = find(strcmpi(headers,'file_system_path'));
while true,
  s = fgetl(fid);
  if ~ischar(s),
    break;
  end
  ss = regexp(s,',','split');
  infocurr = struct;
  for j = 1:numel(ss),
    if j == expdiri,
      expdirstr = regexp(strtrim(ss{j}),'\s+','split');
      if numel(expdirstr) > 1,
        fprintf('Found spaces in file system path %s, using %s.\n',ss{j},expdirstr{1});
      end
      ss{j} = expdirstr{1};
    end
    infocurr.(headers{j}) = ss{j};
  end
  if isempty(infocurr.label),
    continue;
  end
  expinfo = structappend(expinfo,infocurr);
end

fclose(fid);

nexps = numel(expinfo);
for i = 1:nexps,
  if ~exist(expinfo(i).file_system_path,'dir'),
    fprintf('Directory %d %s missing\n',i,expinfo(i).file_system_path);
  end
end

%% make copies of the data here


if ~exist(rootdatadir,'dir'),
  mkdir(rootdatadir);
end

expdirs = cell(1,nexps);

for i = 1:nexps,
  [~,expname] = fileparts(expinfo(i).file_system_path);
  outexpdir = fullfile(rootdatadir,expname);
  if domakecopies && ~exist(outexpdir,'dir'),
    fprintf('Copying %d %s\n',i,expinfo(i).file_system_path);
    expdirs{i} = SymbolicCopyExperimentDirectory(expinfo(i).file_system_path,rootdatadir);
  else
    expdirs{i} = outexpdir;
  end
end

%% read in analysis info if available

loginfos = [];
ispreprocessed = false(1,nexps);
for i = 1:nexps,
  expdir = expdirs{i};
  logfile = fullfile(expdir,'flydisco-analysis-log.txt');
  if exist(logfile,'file'),
    [loginfo,success] = ParseFlyDiscoLog(logfile);
    assert(success);
    [~,ap] = fileparts(loginfo.analysis_protocol_folder);
    ispreprocessed(i) = ismember(ap,isdoneprotocols);
  else
    loginfo = struct;
  end
  loginfos = structappend(loginfos,loginfo);
end

%% track experiments

forcecompute = false;
datalocparamsfile = fullfile(settingsdir,analysis_protocol,datalocparamsfilestr);
dataloc_params = ReadParams(datalocparamsfile);
mintimestamp = datenum('20211013','yyyymmdd');

parentparams = load(fullfile(settingsdir,analysis_protocol,dataloc_params.flytrackerparentcalibrationstr));
fprintf('Parent calibration:\n');
disp(parentparams.calib);
disp(parentparams.calib.params);

params = ReadParams(fullfile(settingsdir,analysis_protocol,dataloc_params.flytrackeroptionsstr));
fprintf('Options:\n');
disp(params);

for i = 1:nexps,
  expdir = expdirs{i};
  [~,expname] = fileparts(expdir);
  
  outfile = fullfile(expdir,'movie-track.mat');
  if exist(outfile,'file'),
    res = dir(outfile);
    if ispreprocessed(i) || res.datenum > mintimestamp,
      fprintf('%s done, skipping\n',expname);
      continue;
    end
  end
  
  matfile = fullfile(rundir,sprintf('flytracker_%s.mat',expname));
  save(matfile,'expdir','settingsdir','analysis_protocol','dataloc_params','forcecompute');
  
  logfile = fullfile(rundir,sprintf('flytracker_%s_%s.out',expname,timestamp));
  resfile = fullfile(rundir,sprintf('flytracker_%s_%s_bsub.out',expname,timestamp));

  cmd = sprintf('ssh login1 "bsub -n4 -J %s -o %s -e %s \\\"cd %s; %s -nodisplay -batch \\\\\\"addpath %s; modpath; FlyDiscoGenericWrapper(''FlyTrackerWrapperForFlyDisco'',''%s''); exit;\\\\\\" > %s 2>&1\\\""',expname,resfile,resfile,cwd,matlabpath,fdapath,matfile,logfile);
  unix(cmd);
  
%   FlyTrackerWrapperForFlyDisco(expdir, settingsdir, analysis_protocol, dataloc_params, forcecompute) ;
%   td = load(fullfile(expdir,'movie_JAABA','trx.mat'));
%   ntraj(i) = numel(td.trx);
%   fprintf('%s: %d\n',expname,ntraj(i));
end

%% collect results

fprintf('N. trajectories per experiment:\n');
ntraj = nan(1,nexps);
nflies = nan(1,nexps);
for i = 1:nexps,
  expdir = expdirs{i};
  [~,expname] = fileparts(expdir);
  
  outfile = fullfile(expdir,'movie-track.mat');
  if ~exist(outfile,'file'),
    continue;
  end
  res = dir(outfile);
  if res.datenum <= mintimestamp,
    continue;
  end
  td = load(outfile);
  nflies(i) = median(sum(~isnan(td.trk.data(:,:,1)),1));
  ntraj(i) = size(td.trk.data,1);
  fprintf('%s: %d flies, %d traj\n',expname,nflies(i),ntraj(i));
end
fprintf('%d / %d experiments tracked\n',nnz(~isnan(ntraj)),nexps);
istracked = ~isnan(nflies);

% for i = find(~istracked),
%   expdir = expdirs{i};
%   [~,expname] = fileparts(expdir);
%   logfile = fullfile(rundir,sprintf('flytracker_%s_%s.out',expname,timestamp));
%   if exist(logfile,'file'),
%     fprintf('%s log file:\n',expname);
%     type(logfile);
%     fprintf('=======\n');
%   else
%     fprintf('No log file for %s\n',expname);
%   end
% end

%% registration

mintimestamp = datenum('20211014','yyyymmdd');

for i = 1:nexps,

  if ~istracked(i),
    continue;
  end

  expdir = expdirs{i};
  [~,expname] = fileparts(expdir);
  istrp = ~isempty(regexp(expname,'nochr_Trp','once'));
  if istrp,
    analysis_protocol_curr = reg_trp_analysis_protocol;
  else
    if strcmpi(expinfo(i).LED_panel,'RED'),
      analysis_protocol_curr = reg_chr_analysis_protocol_red;
    elseif strcmpi(expinfo(i).LED_panel,'RGB'),
      analysis_protocol_curr = reg_chr_analysis_protocol_rgb;
    else
      error('unknown led panel >%s<',expinfo(i).LED_panel);
    end
  end
  
  outfile = fullfile(expdir,'registrationdata.mat');
  if exist(outfile,'file'),
    res = dir(outfile);
    if ispreprocessed(i) || res.datenum > mintimestamp,
      fprintf('%s done, skipping\n',expname);
      continue;
    end
  end
  FlyDiscoRegisterTrx(expdir,'analysis_protocol',analysis_protocol_curr,'settingsdir',reg_settingsdir);
end

isregistered = false(1,nexps);
for i = 1:nexps,
  expdir = expdirs{i};
  outfile = fullfile(expdir,'registrationdata.mat');
  if exist(outfile,'file'),
    res = dir(outfile);
    if res.datenum > mintimestamp,
      isregistered(i) = true;
    end
  end
end

%% indicator LED detection

for i = 1:nexps,

  if ~isregistered(i),
    continue;
  end

  expdir = expdirs{i};
  [~,expname] = fileparts(expdir);
  istrp = ~isempty(regexp(expname,'nochr_Trp','once'));
  if istrp,
    continue;
  end
  if strcmpi(expinfo(i).LED_panel,'RED'),
    analysis_protocol_curr = reg_chr_analysis_protocol_red;
  elseif strcmpi(expinfo(i).LED_panel,'RGB'),
    analysis_protocol_curr = reg_chr_analysis_protocol_rgb;
  else
    error('unknown led panel >%s<',expinfo(i).LED_panel);
  end
    
  outfile = fullfile(expdir,'indicatordata.mat');
  if exist(outfile,'file'),
    res = dir(outfile);
    if ispreprocessed(i) || res.datenum > mintimestamp,
      fprintf('%s done, skipping\n',expname);
      continue;
    end
  end
  %FlyDiscoDectectIndicatorLedOnOff(expdir,'analysis_protocol',analysis_protocol_curr,'settingsdir',reg_settingsdir);
  
  logfile = fullfile(rundir,sprintf('indicator_%s_%s.out',expname,timestamp));
  resfile = fullfile(rundir,sprintf('indicator_%s_%s_bsub.out',expname,timestamp));

  cmd = sprintf('ssh login1 "bsub -n2 -J %s -o %s -e %s \\\"cd %s; %s -nodisplay -batch \\\\\\"addpath %s; modpath; FlyDiscoDectectIndicatorLedOnOff(''%s'',''settingsdir'',''%s'',''analysis_protocol'',''%s''); exit;\\\\\\" > %s 2>&1\\\""',...
    expname,resfile,resfile,cwd,matlabpath,fdapath,expdir,reg_settingsdir,analysis_protocol_curr,logfile);
  unix(cmd);
  
end

isindicator = false(1,nexps);
for i = 1:nexps,
  expdir = expdirs{i};
  [~,expname] = fileparts(expdir);
  istrp = ~isempty(regexp(expname,'nochr_Trp','once'));
  if istrp,
    isindicator(i) = true;
    continue;
  end
  outfile = fullfile(expdir,'indicatordata.mat');
  if exist(outfile,'file'),
    res = dir(outfile);
    if res.datenum > mintimestamp,
      isindicator(i) = true;
    end
  end
end

%% sex classification

for i = 1:nexps,

  if ~isregistered(i),
    continue;
  end
  
%   if isclassifiedsex(i),
%     continue;
%   end
  
  expdir = expdirs{i};
  [~,expname] = fileparts(expdir);
  
  istrp = ~isempty(regexp(expname,'nochr_Trp','once'));
  metadata = ReadMetadataFile(fullfile(expdir,'Metadata.xml'));
  dofix = ~istrp && strcmpi(metadata.gender,'b');
  
  outfile = fullfile(expdir,dataloc_params.sexclassifierdiagnosticsfilestr);
  regfile = fullfile(expdir,'registered_trx.mat');

  if ~dofix && exist(outfile,'file') && exist(regfile,'file'),
    [outfile_src,oislink] = GetLinkSources({outfile});
    [regfile_src,rislink] = GetLinkSources({regfile});
    outfile_src = outfile_src{1};
    regfile_src = regfile_src{1};
    od = dir(outfile_src);
    rd = dir(regfile_src);
    if (~oislink || rislink) && abs(rd.datenum-od.datenum) <= 1/24,
      fprintf('Sex classification complete for %d %s, skipping\n',i,expname);
      continue;
    end
  end
  
  fprintf('Classifying sex for %d %s...\n',i,expname);
  
  logfile = fullfile(rundir,sprintf('classifysex_%s_%s.out',expname,timestamp));
  resfile = fullfile(rundir,sprintf('classifysex_%s_%s_bsub.out',expname,timestamp));

  if dofix,
    FlyDiscoClassifySex(expdir,'settingsdir',settingsdir,'analysis_protocol',analysis_protocol,'override_gender','f');
  else
    cmd = sprintf('ssh login1 "bsub -n2 -J %s -o %s -e %s \\\"cd %s; %s -nodisplay -batch \\\\\\"addpath %s; modpath; FlyDiscoClassifySex(''%s'',''settingsdir'',''%s'',''analysis_protocol'',''%s''); exit;\\\\\\" > %s 2>&1\\\""',...
      expname,resfile,resfile,cwd,matlabpath,fdapath,expdir,reg_settingsdir,analysis_protocol,logfile);
    unix(cmd);
    %disp(cmd);
  end
  
end

metadata_gender = repmat('?',[1,nexps]);
isclassifiedsex = false(1,nexps);
for i = 1:nexps,

  if ~isregistered(i),
    continue;
  end
  
  expdir = expdirs{i};
  [~,expname] = fileparts(expdir);

  metadata = ReadMetadataFile(fullfile(expdir,'Metadata.xml'));
  metadata_gender(i) = metadata.gender;
  
  outfile = fullfile(expdir,dataloc_params.sexclassifierdiagnosticsfilestr);
  regfile = fullfile(expdir,'registered_trx.mat');

  if exist(outfile,'file') && exist(regfile,'file'),
    [outfile_src,oislink] = GetLinkSources({outfile});
    [regfile_src,rislink] = GetLinkSources({regfile});
    outfile_src = outfile_src{1};
    regfile_src = regfile_src{1};
    od = dir(outfile_src);
    rd = dir(regfile_src);
    if (~oislink || rislink) && abs(rd.datenum-od.datenum) <= 1/24,
      isclassifiedsex(i) = true;
    else
      fprintf('%d %s\n',i,expname);
    end
  end
  
end

%% choose videos to track with apt

ntraj_reg = nan(1,nexps);
for i = 1:nexps,
  if ~isregistered(i),
    continue;
  end
  td = load(fullfile(expdirs{i},'registered_trx.mat'));
  ntraj_reg(i) = numel(td.trx);
end

nmales = nan(1,nexps);
nfemales = nan(1,nexps);
for i = 1:nexps,
  if ~isclassifiedsex(i),
    continue;
  end
  expdir = expdirs{i};
  sp = ReadParams(fullfile(expdir,'sexclassifier_diagnostics.txt'));
  nmales(i) = sp.median_nmales;
  nfemales(i) = sp.median_nfemales;
end

for i = 1:nexps,
  expdir = expdirs{i};
  [~,expname] = fileparts(expdir);
  fprintf('%s: %d flies, fly tracker n. trajs, %d reg n. trajs %d\n',expname,nflies(i),ntraj(i),ntraj_reg(i));
end
% 
% for i = 1:nexps,
%   fprintf('%d,%d\n',nflies(i),ntraj_reg(i));
% end


minnflies = 9;
doapt = isindicator & nflies >= minnflies;

% how many videos per condition
[labels,~,labelidx] = unique({expinfo(doapt).label});
counts = hist(labelidx,1:numel(labels));
for i = 1:numel(labels),
  fprintf('%s: %d\n',labels{i},counts(i));
end

% how much time for each condition? 
frspercondition = zeros(1,numel(labels));
fliespercondition = zeros(1,numel(labels));
aptidx = find(doapt);
for i = 1:numel(aptidx),
  expdir = expdirs{aptidx(i)};
  trxfile = fullfile(expdir,'registered_trx.mat');
  td = load(trxfile);
  frspercondition(labelidx(i)) = frspercondition(labelidx(i)) + sum(([td.trx.endframe]-[td.trx.firstframe])+1);
  fliespercondition(labelidx(i)) = fliespercondition(labelidx(i)) + numel(td.trx);
end
fps = 150;
for i = 1:numel(labels),
  fprintf('%s: %d videos, %d flies, %f minutes\n',labels{i},counts(i),fliespercondition(i),frspercondition(i)/fps/50);
end

% 20A02: 11 videos, 115 flies, 594.121867 minutes
% 65F12: 12 videos, 168 flies, 707.778000 minutes
% 71G01: 13 videos, 133 flies, 822.107467 minutes
% 91B01: 10 videos, 95 flies, 570.106267 minutes
% BDP_sexseparated: 4 videos, 38 flies, 228.045067 minutes
% BlindControl: 11 videos, 107 flies, 836.369733 minutes
% BlindaIPgControl: 7 videos, 71 flies, 555.058400 minutes
% BlindpC1dControl: 7 videos, 69 flies, 539.473867 minutes
% Control_RGB: 7 videos, 70 flies, 547.194000 minutes
% MixedaIPgBlind: 4 videos, 37 flies, 289.074667 minutes
% MixedaIPgBlindEmpty: 6 videos, 59 flies, 445.640533 minutes
% MixedaIPgEmpty: 5 videos, 48 flies, 351.821333 minutes
% aIPgBlind_newstim: 11 videos, 108 flies, 844.212933 minutes
% aIPgpublished1: 3 videos, 31 flies, 168.209467 minutes
% aIPgpublished2: 2 videos, 19 flies, 114.149067 minutes
% aIPgpublished3_newstim: 10 videos, 97 flies, 734.849867 minutes
% aIPgunpublished: 2 videos, 19 flies, 114.144533 minutes
% control: 9 videos, 83 flies, 498.092533 minutes
% control_RED: 2 videos, 23 flies, 114.183600 minutes
% male20A02_femaleBDP: 4 videos, 39 flies, 228.060000 minutes
% male65F12_femaleBDP: 2 videos, 25 flies, 113.905333 minutes
% male71G01_femaleBDP: 5 videos, 64 flies, 293.603067 minutes
% pC1dBlind_newstim: 11 videos, 108 flies, 844.395733 minutes
% pC1dpublished1: 3 videos, 28 flies, 168.221067 minutes
% pC1dpublished1_newstim: 8 videos, 80 flies, 625.497333 minutes
% pC1dpublished2: 3 videos, 28 flies, 168.208933 minutes
% pC1epublished: 2 videos, 18 flies, 108.135600 minutes

%% get APT tracker

if docopyapttracker,
  
  addpath(aptpath);
  APT.setpath;
  lObj = StartAPT;
  lObj.projLoad(lblfile);
  
  lObj.printAllTrackerInfo(true);
  
  % Tracker 1: mdn, view 0, mode multiAnimalTDPoseTrx
  %   Trained 20210523T191514 for 60000 iterations on 7208 labels
  %   Stripped lbl file: /groups/branson/home/bransonk/.apt/tp134a8e47_d4c1_40ee_9ae3_c82f377a72d5/multitarget_bubble_training_20210523/20210523T191101_20210523T191514.lbl
  %   Current trained model: /groups/branson/home/bransonk/.apt/tp134a8e47_d4c1_40ee_9ae3_c82f377a72d5/multitarget_bubble_training_20210523/mdn/view_0/20210523T191101/deepnet-60000.index
  % Tracker 3: mdn_joint_fpn, view 0, mode multiAnimalTDPoseTrx
  %   Trained 20210920T192630 for 60000 iterations on 7208 labels
  %   Stripped lbl file: /groups/branson/home/bransonk/.apt/tp134a8e47_d4c1_40ee_9ae3_c82f377a72d5/multitarget_bubble_training_20210523_allGT_AR_params20210920/20210920T192629_20210920T192630.lbl
  %   Current trained model: /groups/branson/home/bransonk/.apt/tp134a8e47_d4c1_40ee_9ae3_c82f377a72d5/multitarget_bubble_training_20210523_allGT_AR_params20210920/mdn_joint_fpn/view_0/20210920T192629/deepnet-60000
  unix(['cp -r /groups/branson/home/bransonk/.apt/tp134a8e47_d4c1_40ee_9ae3_c82f377a72d5 ',apttrackerdir]);
  
end

apttracker = struct;
apttracker.name = '20210920T192629';
apttracker.view = 1;
apttracker.cache = apttrackerdir;
apttracker.model = fullfile(apttracker.cache,'multitarget_bubble_training_20210523_allGT_AR_params20210920/mdn_joint_fpn/view_0/20210920T192629/deepnet-60000');
apttracker.type = 'mdn_joint_fpn';
apttracker.strippedlblfile = fullfile(apttracker.cache,'multitarget_bubble_training_20210523_allGT_AR_params20210920/20210920T192629_20210920T192630.lbl');

save APTTrackerInfo20211108.mat apttracker apttrackerdir aptpath bindpaths singularityimg

assert(exist(apttracker.model,'file')>0);
assert(exist(apttracker.strippedlblfile,'file')>0);

%% APT track

binpathstr = sprintf(' -B %s',bindpaths{:});

for i = 1:nexps,

  if ~doapt(i),
    continue;
  end
  
  expdir = expdirs{i};
  [~,expname] = fileparts(expdir);

  errfile = fullfile(rundir,sprintf('apt_%s_%s.err',expname,timestamp));
  apttrkfile = fullfile(expdir,'apttrk.mat');
  movfile = fullfile(expdir,'movie.ufmf');
  trxfile = fullfile(expdir,'registered_trx.mat');
  
  if exist(apttrkfile,'file'),
    res = dir(apttrkfile);
    if res.datenum > mintimestamp,
      fprintf('%s apt tracked, skipping\n',expname);
      continue;
    end
  end
  
  logfile = fullfile(rundir,sprintf('apt_%s_%s.out',expname,timestamp));
  resfile = fullfile(rundir,sprintf('apt_%s_%s_bsub.out',expname,timestamp));
  reserrfile = fullfile(rundir,sprintf('apt_%s_%s_bsub.err',expname,timestamp));
  
  %aptextra = ' -trx_ids 1 -start_frame 1 -end_frame 201';
  aptextra = '';
  aptcmd = sprintf('python %s/deepnet/APT_interface.py -name %s -view %d -cache %s -err_file %s -model_files %s -type %s %s track -out %s -mov %s -trx %s%s > %s 2>&1',...
    aptpath,apttracker.name,apttracker.view,apttracker.cache,errfile,apttracker.model,apttracker.type,apttracker.strippedlblfile,apttrkfile,movfile,trxfile,aptextra,logfile);
  singcmd = sprintf('singularity exec --nv %s %s bash -c "%s"',binpathstr,singularityimg,aptcmd);
  bsubcmd = sprintf('bsub -n 1 -J apt_%s -gpu "num=1" -q gpu_rtx -o %s -e %s -R"affinity[core(1)]" "%s"',expname,resfile,reserrfile,strrep(singcmd,'"','\"'));
  
  sshcmd = sprintf('ssh login1 "%s"',strrep(strrep(bsubcmd,'\','\\'),'"','\"'));
  
  fprintf('APT tracking %d %s:\n%s\n',i,expname,sshcmd);
  unix(sshcmd);
  
end

%%

isapttracked = false(1,nexps);
for i = 1:nexps,
  expdir = expdirs{i};
  [~,expname] = fileparts(expdir);
  outfile = fullfile(expdir,'apttrk.mat');
  if exist(outfile,'file'),
    res = dir(outfile);
    if res.datenum > mintimestamp,
      isapttracked(i) = true;
    end
  end
end

nnz(isapttracked)

%% read info about activation strengths

activationinfo = [];

for expi = 1:nexps,
  if ~isindicator(expi),
    aic = struct;
    activationinfo = structappend(activationinfo,aic);
    continue;
  end
  expdir = expdirs{expi};
  [~,expname] = fileparts(expdir);
  istrp = ~isempty(regexp(expname,'nochr_Trp','once'));
  if istrp,
    aic = struct;
    activationinfo = structappend(activationinfo,aic);
    continue;
  end
  indd = load(fullfile(expdir,'indicatordata.mat'));
  pd = load(fullfile(expdir,'protocol.mat'));
  nperiods = numel(indd.indicatorLED.startframe);
  aic = struct;
  aic.intensities = nan(1,nperiods);
  aic.pulsewidths = nan(1,nperiods);
  aic.pulseperiods = nan(1,nperiods);
  j = 0;

  aic.LED_panel = expinfo(expi).LED_panel;

  if strcmpi(expinfo(expi).LED_panel,'RGB'),
    intensityfn = 'Rintensity';
    pulsewidthfn = 'RpulseWidth';
    pulseperiodfn = 'RpulsePeriod';
    iterationfn = 'Riteration';
  else
    intensityfn = 'intensity';
    pulsewidthfn = 'pulseWidthSP';
    pulseperiodfn = 'pulsePeriodSP';
    iterationfn = 'iteration';
  end
  
  for i = 1:numel(pd.protocol.(intensityfn)),
    aic.intensities(j+1:j+pd.protocol.(iterationfn)(i)) = pd.protocol.(intensityfn)(i);
    aic.pulsewidths(j+1:j+pd.protocol.(iterationfn)(i)) = pd.protocol.(pulsewidthfn)(i);
    aic.pulseperiods(j+1:j+pd.protocol.(iterationfn)(i)) = pd.protocol.(pulseperiodfn)(i);
    j = j + pd.protocol.(iterationfn)(i);
  end
  activationinfo = structappend(activationinfo,aic);
end

chridx = find(~cellfun(@isempty,{activationinfo.LED_panel}));
for ii = 1:numel(chridx),
  i = chridx(ii);
  activationinfo(i).strengths = activationinfo(i).intensities .* activationinfo(i).pulseperiods ./ activationinfo(i).pulsewidths; %#ok<SAGROW>
end
[LED_panels,~,ledidx] = unique({activationinfo(chridx).LED_panel});

for i = 1:numel(LED_panels),
  idxcurr = chridx(ledidx==i);
  strengths = cat(2,activationinfo(idxcurr).strengths);
  unique_strengths = unique(strengths);
  for jj = 1:numel(idxcurr),
    j = idxcurr(jj);
    [~,strengthidx] = ismember(activationinfo(j).strengths,unique_strengths);
    activationinfo(j).strengthclass = strengthidx./numel(unique_strengths); %#ok<SAGROW>
  end
end


%% collate data from apt and flytracker

landmark_names = {
  'antennae'
  'right_eye'
  'left_eye'
  'left_shoulder'
  'right_shoulder'
  'end_notum'
  'end_abdomen'
  'middle_left_b'
  'middle_left_e'
  'middle_right_b'
  'middle_right_e'
  'tip_front_right'
  'tip_middle_right'
  'tip_back_right'
  'tip_back_left'
  'tip_middle_left'
  'tip_front_left'
  };

unique_labels = unique({expinfo.label});
labels_include = {'control','BDP_sexseparated','Control_RGB','71G01','male71G01_femaleBDP','65F12',...
  '91B01','BlindControl','aIPgpublished3_newstim','pC1dpublished1_newstim','aIPgBlind_newstim'};
assert(all(ismember(labels_include,unique_labels)));

label_superclasses = struct;
label_superclasses.courtship = {'65F12','71G01','male71G01_femaleBDP'};
label_superclasses.control = {'control','Control_RGB','BlindControl','BDP_sexseparated'};
label_superclasses.blind = {'BlindControl','aIPgBlind_newstim'};
label_superclasses.aIPg = {'aIPgBlind_newstim','aIPgpublished3_newstim'};
label_superclasses.aggression = [label_superclasses.aIPg,{'pC1dpublished1_newstim'}];
label_superclasses.GMR71G01 = {'71G01','male71G01_femaleBDP'};
label_superclasses.sexseparated = {'BDP_sexseparated','male71G01_femaleBDP'};
%exporder = randperm(nexps);

nmanuallabels = numel(labeljabfiles);
manualbehaviors = cell(1,nmanuallabels);
jablbldata = cell(1,nmanuallabels);
for i = 1:nmanuallabels,
  jablbldata{i} = loadAnonymous(labeljabfiles{i});
  jablbldata{i}.expNames = cellfun(@fileBaseName,jablbldata{i}.expDirNames,'Uni',0);
  manualbehaviors{i} = jablbldata{i}.behaviors.names{1};
end

yidx = struct;
yidx.light_strength = 1;
yidx.light_period = 2;
yidx.female = 3;
nlabelclasses = numel(labels_include);
superclasses = fieldnames(label_superclasses);
nsuperclasses = numel(superclasses);
nclasses = numel(fieldnames(yidx)) + nlabelclasses + nsuperclasses + nmanuallabels;
ynames = [fieldnames(yidx);labels_include';superclasses;cellfun(@(x) ['perframe_',x],manualbehaviors','Uni',0)];

maxframesfillsex = 300;
isfinaldata = false(1,nexps);

for expii = 1:nexps,
  expi = exporder(expii);
  
  if ~ismember(expinfo(expi).label,labels_include) || ~isapttracked(expi),
    continue;
  end
  
  expdir = expdirs{expi};
  [~,expname] = fileparts(expdir);
  fprintf('Exp %d/%d %s, %s\n',expi,nexps,expname,expinfo(expi).label);
  
  apttrkfile = fullfile(expdir,'apttrk.mat');
  flytrackerfile = fullfile(expdir,'movie-track.mat');
  trxfile = fullfile(expdir,'registered_trx.mat');
  registrationmatfile = fullfile(expdir,'registrationdata.mat');
  
  aptd = TrkFile.load(apttrkfile);
  ftd = load(flytrackerfile);
  rtd = load(trxfile);
  reginfo = load(registrationmatfile);
  reginfo = detectRegistrationMarks('registrationData',reginfo);
  
  [xtmp] = reginfo.registerfn([1,size(reginfo.bkgdImage,2)],reginfo.circleCenterY+[0,0]);
  pxpermm = size(reginfo.bkgdImage,2)/diff(xtmp);
  
  newid2oldid = reginfo.newid2oldid;
  
  nflies = numel(rtd.trx);
  nframes = max([rtd.trx.endframe]);
  npts = aptd.npts;
    
  fidx = struct;
  fidx.wing_left_x = find(strcmp(ftd.trk.names,'wing l x'));
  fidx.wing_left_y = find(strcmp(ftd.trk.names,'wing l y'));
  fidx.wing_right_x = find(strcmp(ftd.trk.names,'wing r x'));
  fidx.wing_right_y = find(strcmp(ftd.trk.names,'wing r y'));
  fidx.body_area = find(strcmp(ftd.trk.names,'body area'));
  fidx.fg_area = find(strcmp(ftd.trk.names,'fg area'));
  fidx.img_contrast = find(strcmp(ftd.trk.names,'img contrast'));
  fidx.min_fg_dist = find(strcmp(ftd.trk.names,'min fg dist'));
  
  
  ndatapts = 13 + 3*npts;
  Xnames = cell(1,ndatapts);
  
  X = nan(nflies,nframes,ndatapts);
  for i = 1:nflies,
    
    id = newid2oldid(i);
    ff = rtd.trx(i).firstframe;
    ef = rtd.trx(i).endframe;
    
    j = 1;
    Xnames{j} = 'x_mm';
    X(i,ff:ef,j) = rtd.trx(i).x_mm;
    j = j+1;
    Xnames{j} = 'y_mm';
    X(i,ff:ef,j) = rtd.trx(i).y_mm;
    j = j+1;
    Xnames{j} = 'ori_rad';
    X(i,ff:ef,j) = rtd.trx(i).theta_mm;
    j = j+1;
    Xnames{j} = 'maj_ax_mm';
    X(i,ff:ef,j) = rtd.trx(i).a_mm*4;
    j = j+1;
    Xnames{j} = 'min_ax_mm';
    X(i,ff:ef,j) = rtd.trx(i).b_mm*4;
    j = j+1;
    
    x = ftd.trk.data(id,ff:ef,fidx.wing_left_x);
    y = ftd.trk.data(id,ff:ef,fidx.wing_left_y);
    [x_mm,y_mm] = reginfo.registerfn(x,y);
    Xnames{j} = 'wing_left_x';
    X(i,ff:ef,j) = x_mm;
    j = j+1;
    Xnames{j} = 'wing_left_y';
    X(i,ff:ef,j) = y_mm;
    j = j+1;
    
    x = ftd.trk.data(id,ff:ef,fidx.wing_right_x);
    y = ftd.trk.data(id,ff:ef,fidx.wing_right_y);
    [x_mm,y_mm] = reginfo.registerfn(x,y);
    Xnames{j} = 'wing_right_x';
    X(i,ff:ef,j) = x_mm;
    j = j+1;
    Xnames{j} = 'wing_right_y';
    X(i,ff:ef,j) = y_mm;
    j = j+1;
    
    Xnames{j} = 'body_area_mm2';
    X(i,ff:ef,j) = ftd.trk.data(id,ff:ef,fidx.body_area)/pxpermm^2;
    j = j+1;
    
    Xnames{j} = 'fg_area_mm2';
    X(i,ff:ef,j) = ftd.trk.data(id,ff:ef,fidx.fg_area)/pxpermm^2;
    j = j+1;
    
    Xnames{j} = 'img_contrast';
    X(i,ff:ef,j) = ftd.trk.data(id,ff:ef,fidx.img_contrast);
    j = j+1;
    
    Xnames{j} = 'min_fg_dist';
    X(i,ff:ef,j) = ftd.trk.data(id,ff:ef,fidx.min_fg_dist)/pxpermm;
    j = j+1;
    
    
    p = aptd.getPTrkTgt(i);
    conf = aptd.getPAuxTgt(i,'pTrkConf');
    for k = 1:npts,
      
      [x_mm,y_mm] = reginfo.registerfn(p(k,1,:),p(k,2,:));
      Xnames{j} = [landmark_names{k},'_x_mm'];
      X(i,:,j) = x_mm;
      j = j+1;
      Xnames{j} = [landmark_names{k},'_y_mm'];
      X(i,:,j) = y_mm;
      j = j+1;
      Xnames{j} = [landmark_names{k},'_conf'];
      X(i,:,j) = conf(k,:)/2;
      j = j+1;
    end
    
  end
  
  istrp = ~isempty(regexp(expname,'nochr_Trp','once'));
  if ~istrp,
    indd = load(fullfile(expdir,'indicatordata.mat'));
    nperiods = numel(indd.indicatorLED.startframe);
    activation_period = zeros(1,nframes);
    for i = 1:nperiods,
      sf = indd.indicatorLED.startframe(i);
      ef = indd.indicatorLED.endframe(i);
      activation_period(sf:ef) = i;
    end
    aic = activationinfo(expi);
    activation_strength = zeros(1,nframes);
    activation_strength(activation_period > 0) = aic.strengthclass(activation_period(activation_period>0));
    
  end
  
  y = nan(nflies,nframes,nclasses);
  
  expidx = nan(1,nmanuallabels);
  for k = 1:nmanuallabels,
    
    expidxcurr = find(strcmp(expname,jablbldata{k}.expNames));
    assert(numel(expidxcurr)<=1);
    if ~isempty(expidxcurr),
      expidx(k) = expidxcurr;
    end
    
  end
  
  for fly = 1:nflies,
    ff = rtd.trx(fly).firstframe;
    ef = rtd.trx(fly).endframe;
    
    if istrp,
      y(fly,ff:ef,yidx.light_strength) = 0;
      y(fly,ff:ef,yidx.light_period) = 0;
    else
      y(fly,ff:ef,yidx.light_strength) = activation_strength(ff:ef);
      y(fly,ff:ef,yidx.light_period) = activation_period(ff:ef) / max(1,max(activation_period(ff:ef)));
    end
    
    isfemale = strcmpi(rtd.trx(fly).sex,'F');
    majfemale = nnz(isfemale) > (ef-ff+1)/2;
    ischange = isfemale(1:end-1)~=isfemale(2:end);
    if any(ischange),
      if majfemale,
        [i0,i1] = get_interval_ends(~isfemale);
      else
        [i0,i1] = get_interval_ends(isfemale);
      end
      l = i1-i0;
      for i = 1:numel(i0),
        if l(i) <= maxframesfillsex,
          isfemale(i0(i):i1(i)-1) = majfemale;
        end
      end
      fprintf('%d swaps of sex for fly %d, filled %d of them.\n',numel(i0),fly,nnz(l<=maxframesfillsex));
      if any(l>maxframesfillsex),
        fprintf('Lengths of unfilled intervals: %s\n',mat2str(l(l>maxframesfillsex)));
      end
    end
    
    
    y(fly,ff:ef,yidx.female) = isfemale;
    
    off = numel(fieldnames(yidx));
    for i=1:nlabelclasses,
      y(fly,ff:ef,off+i) = strcmpi(expinfo(expi).label,labels_include{i});
    end
    off = off + nlabelclasses;
    for i = 1:numel(superclasses),
      y(fly,ff:ef,off+i) = ismember(expinfo(expi).label,label_superclasses.(superclasses{i}));
    end
    off = off + nsuperclasses;
    for i = 1:nmanuallabels,
      expj = expidx(i);
      if isnan(expj),
        y(fly,ff:ef,off+i) = nan;
        continue;
      end
      flyj = find(fly == jablbldata{i}.labels(expj).flies);
      if isempty(flyj) || isempty(jablbldata{i}.labels(expj).t0s{flyj}),
        y(fly,ff:ef,off+i) = nan;
        continue;        
      end
      assert(all(jablbldata{i}.labels(expj).t0s{flyj}>=ff) & all(jablbldata{i}.labels(expj).t1s{flyj}<=ef));
      y(fly,:,off+i) = set_interval_ends(jablbldata{i}.labels(expj).t0s{flyj},jablbldata{i}.labels(expj).t1s{flyj},...
        nframes,jablbldata{i}.labels(expj).names{flyj},jablbldata{i}.behaviors.names{1});      
    end
    
  end
  clf;
  yfly = nan(nflies,nframes,nclasses+1);
  yfly(:,:,1:end-1) = y;
  for fly = 1:nflies,
    ff = rtd.trx(fly).firstframe;
    ef = rtd.trx(fly).endframe;
    yfly(fly,ff:ef,end) = fly/nflies;
  end
  %yfly(isnan(yfly)) = -1;
  imagesc(reshape(permute(yfly,[2,1,3]),nflies*nframes,nclasses+1)');
  set(gca,'YTick',1:nclasses+1,'YTickLabel',[fieldnames(yidx);labels_include';superclasses;cellfun(@(x) ['perframe_',x],manualbehaviors','Uni',0);{'fly'}],'TickLabelInterpreter','none');
  colorbar;
  impixelinfo;
  set(gca,'CLim',[-.25,1.001]);
  title(sprintf('%d: %s, %s',expi,expname,expinfo(expi).label),'interpreter','none');
  drawnow;

  savefile = fullfile(expdirs{expi},'data.mat');
  save(savefile,'X','y','Xnames','ynames');
  isfinaldata(expi) = true;

end

%% fail some videos based on tech notes

badexpnames = {
  'NorpA5_JHS_K_85321_RigD_20210923T074612' % bubble not on correctly
  'NorpA_JHS_K_85321_RigD_20210902T075331' % only 5 lights on detections, should have been 6
  };

isbad = false(1,nexps);
for i = 1:nexps,
  [~,expname] = fileparts(expdirs{i});
  isbad(i) = ismember(expname,badexpnames);
end


%% statistics

stats = struct;
stats.X = struct;
stats.y = struct;
stats.X.meanmean = 0;
stats.X.meanstd = 0;
stats.X.mean = 0;
stats.X.std = 0;
stats.X.nframes = 0;
stats.X.meanprctiles = 0;
stats.X.max = -inf;
stats.X.min = inf;
prctiles_compute = [0,1,5,10,25,50,75,90,95,99,100];
stats.X.meanprctile = 0;
stats.nexps = 0;
stats.y.uniquevals = cell(1,nclasses);
stats.y.fracperexp = cell(1,nclasses);

stats.y.counts = cell(1,nclasses);
stats.y.nframes = 0;

for expi = 1:nexps,
  if ~isfinaldata(expi) || isbad(expi),
    continue;
  end
  
  fprintf('%d/%d\n',expi,nexps);
  
  savefile = fullfile(expdirs{expi},'data.mat');
  load(savefile,'X','y');
  
  Xreshape = reshape(X,size(X,1)*size(X,2),size(X,3));
  stats.X.mean = stats.X.mean + nansum(Xreshape,1);
  stats.X.nframes = stats.X.nframes + sum(~isnan(Xreshape),1);
  stats.nexps = stats.nexps + 1;
  yreshape = reshape(y,size(y,1)*size(y,2),size(y,3));
  for k = 1:nclasses,
    stats.y.uniquevals{k} = union(stats.y.uniquevals{k},unique(yreshape(~isnan(yreshape(:,k)),k)));
  end
  
end

stats.X.mean = stats.X.mean ./ stats.X.nframes;
for k = 1:nclasses,
  stats.y.fracperexp{k} = nan(nexps,numel(stats.y.uniquevals{k}));
  stats.y.counts{k} = 0;
end

for expi = 1:nexps,
  if ~isfinaldata(expi) || isbad(expi),
    continue;
  end
  fprintf('%d/%d\n',expi,nexps);
  savefile = fullfile(expdirs{expi},'data.mat');
  load(savefile,'X','y');
  
  Xreshape = reshape(X,size(X,1)*size(X,2),size(X,3));
  stats.X.meanmean = stats.X.meanmean + nanmean(Xreshape,1)/stats.nexps;
  stats.X.meanstd = stats.X.meanstd + nanstd(Xreshape,1,1)/stats.nexps;
  stats.X.meanprctiles = stats.X.meanprctiles + prctile(Xreshape,prctiles_compute,1)/stats.nexps;
  stats.X.mean = stats.X.mean + nansum(Xreshape,1)./stats.X.nframes;
  stats.X.std = stats.X.std + nansum((Xreshape-stats.X.mean).^2,1)./stats.X.nframes;
  stats.X.max = max(stats.X.max,nanmax(Xreshape,[],1));
  stats.X.min = min(stats.X.min,nanmin(Xreshape,[],1));
  
  yreshape = reshape(y,size(y,1)*size(y,2),size(y,3));
  for k = 1:nclasses,
    if all(isnan(yreshape(:,k))),
      continue;
    end
    c = hist(yreshape(~isnan(yreshape(:,k)),k),stats.y.uniquevals{k});
    stats.y.fracperexp{k}(expi,:) = c / nnz(~isnan(yreshape(:,k)));
    stats.y.counts{k} = stats.y.counts{k} + c;
  end
  stats.y.nframes = stats.y.nframes + sum(~isnan(yreshape),1);  
  
end

stats.X.std = sqrt(stats.X.std);

colwidth = 8;
maxnamelength = max(cellfun(@numel,Xnames));
prctileidx = find(prctiles_compute > 0 & prctiles_compute < 100);

fprintf(sprintf('%%%ds ',maxnamelength),'feature');
fprintf(sprintf('| %%%ds ',colwidth),'mean','std','min','max'); %#ok<CTPCT>
fprintf(sprintf('| %%%ddth ',colwidth-2),prctiles_compute(prctileidx));
fprintf('prctile\n');
for i = 1:ndatapts,
  fprintf('%-21s ',Xnames{i});
  fprintf(sprintf('| %% %d.2f ',colwidth),stats.X.mean(i),stats.X.std(i),...
    stats.X.min(i),stats.X.max(i),...
    stats.X.meanprctiles(prctileidx,i));
  fprintf('\n');
end

%               feature |     mean |      std |      min |      max |      1th |      5th |     10th |     25th |     50th |     75th |     90th |     95th |     99th prctile
% x_mm                  |    -1.10 |    13.44 |   -25.50 |    25.43 |   -22.49 |   -20.52 |   -18.37 |   -11.92 |    -0.73 |    10.98 |    17.76 |    20.15 |    22.32 
% y_mm                  |     3.05 |    13.35 |   -24.91 |    25.35 |   -21.98 |   -19.68 |   -17.10 |    -9.69 |     2.09 |    12.95 |    18.95 |    20.94 |    22.74 
% ori_rad               |     0.03 |     1.82 |    -3.14 |     3.14 |    -3.08 |    -2.83 |    -2.51 |    -1.55 |     0.02 |     1.57 |     2.51 |     2.82 |     3.08 
% maj_ax_mm             |     5.48 |     1.55 |     0.80 |     6.82 |     2.16 |     2.41 |     2.48 |     2.58 |     2.73 |     2.91 |     2.98 |     3.02 |     3.10 
% min_ax_mm             |     1.92 |     0.53 |     0.48 |     2.86 |     0.85 |     0.87 |     0.88 |     0.90 |     0.96 |     1.00 |     1.03 |     1.04 |     1.06 
% wing_left_x           |    -1.04 |    13.22 |   -26.59 |    25.92 |   -21.92 |   -20.01 |   -18.06 |   -11.84 |    -0.65 |    10.85 |    17.45 |    19.63 |    21.71 
% wing_left_y           |     3.03 |    13.12 |   -25.87 |    26.60 |   -21.41 |   -19.21 |   -16.85 |    -9.60 |     2.12 |    12.82 |    18.61 |    20.40 |    22.12 
% wing_right_x          |    -1.11 |    13.21 |   -26.59 |    25.91 |   -21.91 |   -20.01 |   -18.06 |   -11.83 |    -0.70 |    10.80 |    17.42 |    19.64 |    21.73 
% wing_right_y          |     3.04 |    13.11 |   -25.92 |    26.39 |   -21.41 |   -19.21 |   -16.81 |    -9.56 |     2.12 |    12.79 |    18.58 |    20.39 |    22.11 
% body_area_mm2         |     4.00 |     1.08 |     0.35 |     5.41 |     1.50 |     1.64 |     1.69 |     1.77 |     1.96 |     2.19 |     2.27 |     2.32 |     2.39 
% fg_area_mm2           |    10.92 |     4.09 |     1.51 |    48.68 |     3.67 |     3.95 |     4.06 |     4.26 |     4.67 |     5.38 |     8.23 |    10.15 |    14.55 
% img_contrast          |     0.67 |     0.21 |     0.06 |     1.04 |     0.25 |     0.27 |     0.28 |     0.30 |     0.33 |     0.37 |     0.41 |     0.43 |     0.47 
% min_fg_dist           |     9.02 |     5.42 |     0.00 |    40.09 |     0.00 |     0.07 |     0.22 |     1.00 |     3.13 |     6.71 |    10.98 |    13.99 |    20.03 
% antennae_x_mm         |    -1.10 |    13.66 |   -26.56 |    26.45 |   -23.17 |   -20.97 |   -18.65 |   -12.03 |    -0.70 |    11.08 |    18.03 |    20.59 |    23.03 
% antennae_y_mm         |     3.07 |    13.57 |   -25.73 |    26.25 |   -22.68 |   -20.10 |   -17.31 |    -9.79 |     2.06 |    13.08 |    19.25 |    21.44 |    23.44 
% antennae_conf         |     2.00 |     0.58 |     0.98 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% right_eye_x_mm        |    -1.10 |    13.58 |   -26.28 |    26.07 |   -22.95 |   -20.80 |   -18.57 |   -11.98 |    -0.70 |    11.05 |    17.93 |    20.41 |    22.77 
% right_eye_y_mm        |     3.06 |    13.49 |   -25.40 |    26.09 |   -22.44 |   -19.96 |   -17.24 |    -9.76 |     2.05 |    13.03 |    19.14 |    21.24 |    23.23 
% right_eye_conf        |     2.00 |     0.58 |     0.99 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% left_eye_x_mm         |    -1.11 |    13.58 |   -26.28 |    26.22 |   -22.94 |   -20.79 |   -18.53 |   -11.98 |    -0.73 |    11.03 |    17.92 |    20.43 |    22.80 
% left_eye_y_mm         |     3.06 |    13.49 |   -25.45 |    25.93 |   -22.46 |   -19.94 |   -17.22 |    -9.76 |     2.07 |    13.04 |    19.14 |    21.25 |    23.20 
% left_eye_conf         |     2.00 |     0.58 |     0.98 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% left_shoulder_x_mm    |    -1.11 |    13.56 |   -26.20 |    26.12 |   -22.87 |   -20.74 |   -18.50 |   -11.97 |    -0.73 |    11.02 |    17.89 |    20.38 |    22.72 
% left_shoulder_y_mm    |     3.06 |    13.46 |   -25.36 |    25.84 |   -22.38 |   -19.89 |   -17.19 |    -9.74 |     2.08 |    13.03 |    19.11 |    21.20 |    23.12 
% left_shoulder_conf    |     2.00 |     0.58 |     0.96 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% right_shoulder_x_mm   |    -1.10 |    13.56 |   -26.22 |    25.99 |   -22.88 |   -20.76 |   -18.54 |   -11.97 |    -0.70 |    11.04 |    17.90 |    20.36 |    22.69 
% right_shoulder_y_mm   |     3.06 |    13.47 |   -25.28 |    26.02 |   -22.36 |   -19.91 |   -17.21 |    -9.74 |     2.06 |    13.02 |    19.11 |    21.19 |    23.15 
% right_shoulder_conf   |     2.00 |     0.58 |     0.98 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% end_notum_x_mm        |    -1.11 |    13.42 |   -25.42 |    25.24 |   -22.41 |   -20.46 |   -18.33 |   -11.92 |    -0.74 |    10.97 |    17.72 |    20.09 |    22.23 
% end_notum_y_mm        |     3.05 |    13.33 |   -24.82 |    25.25 |   -21.90 |   -19.62 |   -17.07 |    -9.68 |     2.09 |    12.95 |    18.92 |    20.87 |    22.65 
% end_notum_conf        |     2.00 |     0.58 |     0.99 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% end_abdomen_x_mm      |    -1.09 |    13.29 |   -26.12 |    25.32 |   -22.04 |   -20.17 |   -18.15 |   -11.86 |    -0.71 |    10.90 |    17.56 |    19.84 |    21.89 
% end_abdomen_y_mm      |     3.02 |    13.19 |   -25.05 |    25.20 |   -21.58 |   -19.38 |   -16.94 |    -9.62 |     2.10 |    12.85 |    18.72 |    20.55 |    22.26 
% end_abdomen_conf      |     2.00 |     0.58 |     0.98 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% middle_left_b_x_mm    |    -1.10 |    13.47 |   -25.86 |    25.60 |   -22.60 |   -20.56 |   -18.41 |   -11.92 |    -0.72 |    11.00 |    17.77 |    20.17 |    22.39 
% middle_left_b_y_mm    |     3.05 |    13.38 |   -24.91 |    25.61 |   -22.08 |   -19.73 |   -17.13 |    -9.69 |     2.07 |    12.95 |    18.98 |    20.97 |    22.86 
% middle_left_b_conf    |     2.00 |     0.58 |     0.99 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% middle_left_e_x_mm    |    -1.09 |    13.46 |   -26.14 |    25.95 |   -22.67 |   -20.53 |   -18.39 |   -11.90 |    -0.72 |    11.01 |    17.76 |    20.14 |    22.43 
% middle_left_e_y_mm    |     3.05 |    13.37 |   -25.14 |    25.97 |   -22.13 |   -19.71 |   -17.12 |    -9.69 |     2.06 |    12.92 |    18.98 |    20.94 |    22.94 
% middle_left_e_conf    |     2.00 |     0.58 |     0.98 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% middle_right_b_x_mm   |    -1.11 |    13.46 |   -25.97 |    25.59 |   -22.58 |   -20.54 |   -18.36 |   -11.93 |    -0.72 |    10.96 |    17.76 |    20.19 |    22.46 
% middle_right_b_y_mm   |     3.05 |    13.37 |   -25.23 |    25.59 |   -22.09 |   -19.71 |   -17.11 |    -9.68 |     2.08 |    12.96 |    18.97 |    20.98 |    22.84 
% middle_right_b_conf   |     2.00 |     0.58 |     0.99 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% middle_right_e_x_mm   |    -1.11 |    13.46 |   -26.31 |    25.90 |   -22.64 |   -20.50 |   -18.33 |   -11.91 |    -0.72 |    10.95 |    17.75 |    20.17 |    22.53 
% middle_right_e_y_mm   |     3.05 |    13.36 |   -25.51 |    25.83 |   -22.14 |   -19.68 |   -17.09 |    -9.68 |     2.08 |    12.95 |    18.94 |    20.95 |    22.90 
% middle_right_e_conf   |     2.00 |     0.58 |     0.98 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% tip_front_right_x_mm  |    -1.06 |    13.72 |   -27.31 |    26.85 |   -23.49 |   -21.08 |   -18.70 |   -12.02 |    -0.67 |    11.13 |    18.10 |    20.73 |    23.34 
% tip_front_right_y_mm  |     3.04 |    13.64 |   -26.49 |    27.11 |   -23.00 |   -20.22 |   -17.39 |    -9.82 |     2.02 |    13.04 |    19.32 |    21.58 |    23.81 
% tip_front_right_conf  |     2.00 |     0.58 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% tip_middle_right_x_mm |    -1.04 |    13.50 |   -27.17 |    27.02 |   -23.18 |   -20.56 |   -18.37 |   -11.88 |    -0.67 |    11.05 |    17.75 |    20.18 |    22.88 
% tip_middle_right_y_mm |     3.02 |    13.42 |   -26.23 |    26.80 |   -22.61 |   -19.76 |   -17.12 |    -9.69 |     2.03 |    12.88 |    18.97 |    21.00 |    23.53 
% tip_middle_right_conf |     2.00 |     0.58 |     0.97 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% tip_back_right_x_mm   |    -1.05 |    13.29 |   -26.58 |    26.57 |   -22.39 |   -20.14 |   -18.11 |   -11.80 |    -0.69 |    10.91 |    17.53 |    19.79 |    22.10 
% tip_back_right_y_mm   |     3.00 |    13.21 |   -25.74 |    26.19 |   -21.85 |   -19.37 |   -16.94 |    -9.61 |     2.06 |    12.79 |    18.70 |    20.52 |    22.67 
% tip_back_right_conf   |     2.00 |     0.58 |     0.82 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% tip_back_left_x_mm    |    -1.07 |    13.29 |   -26.80 |    26.14 |   -22.32 |   -20.10 |   -18.06 |   -11.81 |    -0.70 |    10.85 |    17.52 |    19.86 |    22.28 
% tip_back_left_y_mm    |     3.01 |    13.19 |   -25.90 |    26.66 |   -21.85 |   -19.32 |   -16.88 |    -9.60 |     2.07 |    12.82 |    18.66 |    20.51 |    22.63 
% tip_back_left_conf    |     2.00 |     0.58 |     0.73 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% tip_middle_left_x_mm  |    -1.08 |    13.49 |   -27.16 |    26.79 |   -23.10 |   -20.50 |   -18.27 |   -11.88 |    -0.70 |    10.93 |    17.74 |    20.26 |    23.08 
% tip_middle_left_y_mm  |     3.02 |    13.40 |   -26.39 |    26.87 |   -22.61 |   -19.68 |   -17.05 |    -9.68 |     2.05 |    12.92 |    18.91 |    21.00 |    23.45 
% tip_middle_left_conf  |     2.00 |     0.58 |     0.97 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 
% tip_front_left_x_mm   |    -1.08 |    13.72 |   -27.45 |    27.09 |   -23.47 |   -21.09 |   -18.67 |   -12.01 |    -0.70 |    11.10 |    18.09 |    20.75 |    23.43 
% tip_front_left_y_mm   |     3.04 |    13.63 |   -26.51 |    26.91 |   -23.00 |   -20.21 |   -17.37 |    -9.83 |     2.04 |    13.06 |    19.31 |    21.57 |    23.78 
% tip_front_left_conf   |     2.00 |     0.58 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 |     1.00 

hfig = 2;
figure(hfig);
c = ceil(sqrt(nclasses));
r = ceil(nclasses/c);
hax = createsubplots(r,c,.05);
for k = 1:nclasses,
  bar(hax(k),stats.y.uniquevals{k},stats.y.counts{k},1);
  title(hax(k),ynames{k},'Interpreter','none');
  set(hax(j),'box','off');
  [r1,c1] = ind2sub([r,c],k);
  if c1 == 1,
    ylabel(hax(k),'N. frames');
  end
  if r1 == r,
    xlabel(hax(k),'Value');
  end
  axisalmosttight([],hax(k));
end
delete(hax(nclasses+1:end));
save2pdf('LabelInfo.pdf',hfig);

maxnamelength = max(cellfun(@numel,ynames));
for k = 1:nclasses,
  fprintf(sprintf('%%%ds: ',maxnamelength),ynames{k});
  fprintf(' %.2g',stats.y.uniquevals{k});
  fprintf('\n');
end

%         light_strength:  0 0.5 1
%           light_period:  0 0.17 0.33 0.4 0.5 0.6 0.67 0.8 0.83 1
%                 female:  0 1
%                control:  0 1
%       BDP_sexseparated:  0 1
%            Control_RGB:  0 1
%                  71G01:  0 1
%    male71G01_femaleBDP:  0 1
%                  65F12:  0 1
%                  91B01:  0 1
%           BlindControl:  0 1
% aIPgpublished3_newstim:  0 1
% pC1dpublished1_newstim:  0 1
%      aIPgBlind_newstim:  0 1
%              courtship:  0 1
%                control:  0 1
%                  blind:  0 1
%                   aIPg:  0 1
%             aggression:  0 1
%               GMR71G01:  0 1
%           sexseparated:  0 1
%    perframe_aggression:  0 1
%         perframe_chase:  0 1
%     perframe_courtship:  0 1
%    perframe_highfence2:  0 1
%       perframe_wingext:  0 1

%% save figures

hfig = 1;
figure(hfig);
cm = jet(64);
for expi = 1:nexps,

  if ~isfinaldata(expi) || isbad(expi),
    continue;
  end
  
  expname = fileBaseName(expdirs{expi});
  savefile = fullfile(expdirs{expi},'data.mat');
  load(savefile,'X','y');
  nflies = size(X,1);
  nframes = size(X,2);
  
  clf(hfig);
  hax = [axes('Parent',hfig,'Position',[.1,.3050,.875,.645])
    axes('Parent',hfig,'Position',[.1,.05,.875,.245])];

  t = repmat(linspace(-3,3,nframes),[1,nflies]);
  im = [((reshape(permute(X,[2,1,3]),nflies*nframes,ndatapts)-stats.X.mean)./stats.X.std)';t];
  imrgb = colormap_image(im,cm,[-3,3]);
  imrgb(isnan(repmat(im,[1,1,3]))) = 0;
  
  imagesc(hax(1),imrgb);
  set(hax(1),'YTick',1:ndatapts+1,'YTickLabel',[Xnames,{'Time'}],'TickLabelInterpreter','none','XtickLabel',[]);
  hcb = colorbar(hax(1));
  hcb.Label.String = 'Stds';
  title(hax(1),sprintf('%s, %s',expname,expinfo(expi).label),'interpreter','none');
  set(hax(1),'CLim',[-3,3]);
  colormap(hax(1),jet(256));
  
  yfly = nan(nflies,nframes,nclasses+1);
  yfly(:,:,1:end-1) = y;
  for fly = 1:nflies,
    ff = find(~isnan(X(fly,:,1)),1);
    ef = find(~isnan(X(fly,:,1)),1,'last');
    yfly(fly,ff:ef,end) = fly/nflies;
  end
  %yfly(isnan(yfly)) = -1;
  im = reshape(permute(yfly,[2,1,3]),nflies*nframes,nclasses+1)';
  imrgb = colormap_image(im,cm,[0,1]);
  imrgb(isnan(repmat(im,[1,1,3]))) = 0;
  image(hax(2),imrgb);
  set(hax(2),'YTick',1:nclasses+1,'YTickLabel',[fieldnames(yidx);labels_include';superclasses;cellfun(@(x) ['perframe_',x],manualbehaviors','Uni',0);{'fly'}],'TickLabelInterpreter','none','XtickLabel',[]);
  colorbar(hax(2));
  set(hax(2),'CLim',[-.25,1.001]);
  colormap(hax(2),kjet(256));
  save2png(fullfile(expdirs{expi},'DataPlot.png'),hfig);
  
end

%% double check that jab file label counts make sense

jabinfo = cell(1,nmanuallabels);
for k = 1:nmanuallabels,
  jabinfo{k} = getJAABALabelStats(expinfo,[],labeljabfiles{k});
end

% 65F12                  aggression:	 0/12 exps,   0 flies,      0 bouts,      0 frames
% 65F12                  None      :	 2/12 exps,   4 flies,      5 bouts,    250 frames
% 71G01                  aggression:	 1/13 exps,   2 flies,      2 bouts,     49 frames
% 71G01                  None      :	 2/13 exps,   6 flies,      6 bouts,    265 frames
% 91B01                  aggression:	 0/10 exps,   0 flies,      0 bouts,      0 frames
% 91B01                  None      :	 2/10 exps,   5 flies,      6 bouts,    156 frames
% BDP_sexseparated       aggression:	 1/ 4 exps,   1 flies,      1 bouts,     17 frames
% BDP_sexseparated       None      :	 1/ 4 exps,   2 flies,      2 bouts,     46 frames
% BlindControl           aggression:	 2/ 9 exps,   4 flies,      4 bouts,     28 frames
% BlindControl           None      :	 3/ 9 exps,   5 flies,      5 bouts,    172 frames
% Control_RGB            aggression:	 3/ 6 exps,   5 flies,      5 bouts,    155 frames
% Control_RGB            None      :	 3/ 6 exps,   6 flies,      6 bouts,    179 frames
% aIPgBlind_newstim      aggression:	 4/11 exps,  12 flies,     13 bouts,    250 frames
% aIPgBlind_newstim      None      :	 5/11 exps,  13 flies,     16 bouts,    484 frames
% aIPgpublished3_newstim aggression:	 4/ 9 exps,  37 flies,     38 bouts,   1811 frames
% aIPgpublished3_newstim None      :	 4/ 9 exps,  37 flies,     39 bouts,   1313 frames
% control                aggression:	 0/ 9 exps,   0 flies,      0 bouts,      0 frames
% control                None      :	 2/ 9 exps,   6 flies,      7 bouts,    326 frames
% male71G01_femaleBDP    aggression:	 0/ 5 exps,   0 flies,      0 bouts,      0 frames
% male71G01_femaleBDP    None      :	 1/ 5 exps,   3 flies,      3 bouts,     57 frames
% pC1dpublished1_newstim aggression:	 3/ 8 exps,  30 flies,     32 bouts,   1493 frames
% pC1dpublished1_newstim None      :	 3/ 8 exps,  30 flies,     33 bouts,    995 frames

% 65F12                  chase:	 4/12 exps,  14 flies,     77 bouts,  14798 frames
% 65F12                  None :	 4/12 exps,  18 flies,    133 bouts,  21287 frames
% 71G01                  chase:	 3/13 exps,  13 flies,     64 bouts,  14559 frames
% 71G01                  None :	 3/13 exps,  16 flies,    136 bouts,  21249 frames
% 91B01                  chase:	 0/10 exps,   0 flies,      0 bouts,      0 frames
% 91B01                  None :	 3/10 exps,  10 flies,     92 bouts,  16373 frames
% BDP_sexseparated       chase:	 3/ 4 exps,   8 flies,     28 bouts,  10667 frames
% BDP_sexseparated       None :	 3/ 4 exps,   9 flies,     70 bouts,  12271 frames
% BlindControl           chase:	 1/ 9 exps,   1 flies,      2 bouts,    137 frames
% BlindControl           None :	 3/ 9 exps,   8 flies,     95 bouts,  18743 frames
% Control_RGB            chase:	 2/ 6 exps,   3 flies,      4 bouts,    684 frames
% Control_RGB            None :	 3/ 6 exps,   9 flies,     46 bouts,   9536 frames
% aIPgBlind_newstim      chase:	 0/11 exps,   0 flies,      0 bouts,      0 frames
% aIPgBlind_newstim      None :	 3/11 exps,  13 flies,     76 bouts,  15281 frames
% aIPgpublished3_newstim chase:	 3/ 9 exps,   8 flies,     14 bouts,   1900 frames
% aIPgpublished3_newstim None :	 3/ 9 exps,   9 flies,     45 bouts,   6628 frames
% control                chase:	 0/ 9 exps,   0 flies,      0 bouts,      0 frames
% control                None :	 3/ 9 exps,  11 flies,     36 bouts,  10009 frames
% male71G01_femaleBDP    chase:	 3/ 5 exps,  12 flies,     78 bouts,  21417 frames
% male71G01_femaleBDP    None :	 3/ 5 exps,  15 flies,    119 bouts,  17099 frames
% pC1dpublished1_newstim chase:	 3/ 8 exps,   7 flies,     17 bouts,   3352 frames
% pC1dpublished1_newstim None :	 3/ 8 exps,   9 flies,     51 bouts,   7634 frames

% 65F12                  courtship:	 4/12 exps,   7 flies,     10 bouts,    590 frames
% 65F12                  None     :	 4/12 exps,  10 flies,     14 bouts,    687 frames
% 71G01                  courtship:	 4/13 exps,   8 flies,     12 bouts,    702 frames
% 71G01                  None     :	 4/13 exps,  11 flies,     16 bouts,    799 frames
% 91B01                  courtship:	 0/10 exps,   0 flies,      0 bouts,      0 frames
% 91B01                  None     :	 5/10 exps,  20 flies,     30 bouts,   1828 frames
% BDP_sexseparated       courtship:	 2/ 4 exps,   4 flies,     11 bouts,    394 frames
% BDP_sexseparated       None     :	 3/ 4 exps,   9 flies,     10 bouts,    604 frames
% BlindControl           courtship:	 0/ 9 exps,   0 flies,      0 bouts,      0 frames
% BlindControl           None     :	 4/ 9 exps,  12 flies,     15 bouts,    923 frames
% Control_RGB            courtship:	 0/ 6 exps,   0 flies,      0 bouts,      0 frames
% Control_RGB            None     :	 4/ 6 exps,  10 flies,     12 bouts,    939 frames
% aIPgBlind_newstim      courtship:	 0/11 exps,   0 flies,      0 bouts,      0 frames
% aIPgBlind_newstim      None     :	 4/11 exps,  16 flies,     20 bouts,   1542 frames
% aIPgpublished3_newstim courtship:	 0/ 9 exps,   0 flies,      0 bouts,      0 frames
% aIPgpublished3_newstim None     :	 6/ 9 exps,  19 flies,     34 bouts,   2468 frames
% control                courtship:	 0/ 9 exps,   0 flies,      0 bouts,      0 frames
% control                None     :	 4/ 9 exps,  14 flies,     24 bouts,   1203 frames
% male71G01_femaleBDP    courtship:	 3/ 5 exps,   9 flies,     12 bouts,    535 frames
% male71G01_femaleBDP    None     :	 3/ 5 exps,   8 flies,     11 bouts,    456 frames
% pC1dpublished1_newstim courtship:	 0/ 8 exps,   0 flies,      0 bouts,      0 frames
% pC1dpublished1_newstim None     :	 2/ 8 exps,   8 flies,      9 bouts,    701 frames

% 65F12                  highfence2:	 1/12 exps,   1 flies,      1 bouts,      2 frames
% 65F12                  None      :	 6/12 exps,  18 flies,     22 bouts,    474 frames
% 71G01                  highfence2:	 4/13 exps,   7 flies,      7 bouts,     27 frames
% 71G01                  None      :	 4/13 exps,  10 flies,     15 bouts,    190 frames
% 91B01                  highfence2:	 0/10 exps,   0 flies,      0 bouts,      0 frames
% 91B01                  None      :	 4/10 exps,   9 flies,     12 bouts,    119 frames
% BDP_sexseparated       highfence2:	 0/ 4 exps,   0 flies,      0 bouts,      0 frames
% BDP_sexseparated       None      :	 0/ 4 exps,   0 flies,      0 bouts,      0 frames
% BlindControl           highfence2:	 4/ 9 exps,   7 flies,      7 bouts,     40 frames
% BlindControl           None      :	 4/ 9 exps,  21 flies,     25 bouts,    420 frames
% Control_RGB            highfence2:	 1/ 6 exps,   2 flies,      2 bouts,      7 frames
% Control_RGB            None      :	 3/ 6 exps,   8 flies,      8 bouts,    153 frames
% aIPgBlind_newstim      highfence2:	 4/11 exps,  10 flies,     12 bouts,     80 frames
% aIPgBlind_newstim      None      :	 4/11 exps,  16 flies,     20 bouts,    263 frames
% aIPgpublished3_newstim highfence2:	 5/ 9 exps,  26 flies,     43 bouts,    264 frames
% aIPgpublished3_newstim None      :	 5/ 9 exps,  17 flies,     23 bouts,    420 frames
% control                highfence2:	 1/ 9 exps,   1 flies,      1 bouts,      4 frames
% control                None      :	 5/ 9 exps,  14 flies,     18 bouts,    412 frames
% male71G01_femaleBDP    highfence2:	 1/ 5 exps,   1 flies,      1 bouts,      6 frames
% male71G01_femaleBDP    None      :	 1/ 5 exps,   5 flies,      6 bouts,    119 frames
% pC1dpublished1_newstim highfence2:	 5/ 8 exps,  24 flies,     37 bouts,    223 frames
% pC1dpublished1_newstim None      :	 5/ 8 exps,  14 flies,     15 bouts,    224 frames

% 65F12                  wingext:	 3/12 exps,   7 flies,     25 bouts,   2342 frames
% 65F12                  None   :	 3/12 exps,  12 flies,     43 bouts,   4082 frames
% 71G01                  wingext:	 3/13 exps,  10 flies,     64 bouts,   6585 frames
% 71G01                  None   :	 3/13 exps,  14 flies,    127 bouts,   9015 frames
% 91B01                  wingext:	 0/10 exps,   0 flies,      0 bouts,      0 frames
% 91B01                  None   :	 3/10 exps,   6 flies,     26 bouts,   1368 frames
% BDP_sexseparated       wingext:	 3/ 4 exps,   7 flies,     34 bouts,   2496 frames
% BDP_sexseparated       None   :	 3/ 4 exps,  15 flies,     81 bouts,   5790 frames
% BlindControl           wingext:	 0/ 9 exps,   0 flies,      0 bouts,      0 frames
% BlindControl           None   :	 3/ 9 exps,   9 flies,     46 bouts,   4863 frames
% Control_RGB            wingext:	 0/ 6 exps,   0 flies,      0 bouts,      0 frames
% Control_RGB            None   :	 3/ 6 exps,  12 flies,     66 bouts,   7064 frames
% aIPgBlind_newstim      wingext:	 0/11 exps,   0 flies,      0 bouts,      0 frames
% aIPgBlind_newstim      None   :	 3/11 exps,   7 flies,     48 bouts,   3690 frames
% aIPgpublished3_newstim wingext:	 0/ 9 exps,   0 flies,      0 bouts,      0 frames
% aIPgpublished3_newstim None   :	 3/ 9 exps,  14 flies,     48 bouts,   5076 frames
% control                wingext:	 1/ 9 exps,   1 flies,      1 bouts,     20 frames
% control                None   :	 3/ 9 exps,  13 flies,     44 bouts,   3698 frames
% male71G01_femaleBDP    wingext:	 3/ 5 exps,   9 flies,     52 bouts,   4678 frames
% male71G01_femaleBDP    None   :	 3/ 5 exps,  13 flies,     91 bouts,   5801 frames
% pC1dpublished1_newstim wingext:	 0/ 8 exps,   0 flies,      0 bouts,      0 frames
% pC1dpublished1_newstim None   :	 3/ 8 exps,  12 flies,     44 bouts,   3871 frames

% 65F12                  wingflick:	 5/12 exps,  10 flies,     49 bouts,    121 frames
% 65F12                  None     :	 5/12 exps,  21 flies,     37 bouts,    705 frames
% 71G01                  wingflick:	 5/13 exps,  13 flies,     62 bouts,    138 frames
% 71G01                  None     :	 5/13 exps,  22 flies,     34 bouts,    596 frames
% 91B01                  wingflick:	 0/10 exps,   0 flies,      0 bouts,      0 frames
% 91B01                  None     :	 4/10 exps,  18 flies,     43 bouts,   1001 frames
% BDP_sexseparated       wingflick:	 3/ 4 exps,   7 flies,     41 bouts,    129 frames
% BDP_sexseparated       None     :	 4/ 4 exps,  13 flies,     31 bouts,    421 frames
% BlindControl           wingflick:	 3/ 9 exps,   6 flies,     13 bouts,     30 frames
% BlindControl           None     :	 3/ 9 exps,   8 flies,     21 bouts,    554 frames
% Control_RGB            wingflick:	 3/ 6 exps,   8 flies,     15 bouts,     41 frames
% Control_RGB            None     :	 3/ 6 exps,   8 flies,     16 bouts,    399 frames
% aIPgBlind_newstim      wingflick:	 2/11 exps,   3 flies,     19 bouts,     41 frames
% aIPgBlind_newstim      None     :	 4/11 exps,  12 flies,     23 bouts,    653 frames
% aIPgpublished3_newstim wingflick:	 4/ 9 exps,  14 flies,     34 bouts,     88 frames
% aIPgpublished3_newstim None     :	 4/ 9 exps,  15 flies,     24 bouts,    487 frames
% control                wingflick:	 1/ 9 exps,   1 flies,      1 bouts,      2 frames
% control                None     :	 4/ 9 exps,  11 flies,     16 bouts,    331 frames
% male71G01_femaleBDP    wingflick:	 2/ 5 exps,   5 flies,     24 bouts,     52 frames
% male71G01_femaleBDP    None     :	 2/ 5 exps,   9 flies,     19 bouts,    335 frames
% pC1dpublished1_newstim wingflick:	 4/ 8 exps,  12 flies,     34 bouts,     83 frames
% pC1dpublished1_newstim None     :	 4/ 8 exps,  12 flies,     28 bouts,    630 frames

%% figure out some training / test1 / test2 splits

% goals:
% have enough of each label type in each split
% keep whole videos in one split if possible
% if not possible, keep a large buffer between frames included in training
% vs test splits

finalexpidx = find(isfinaldata & ~isbad); % row vec

% load all labels
ally = cell(1,nexps);

for expi = finalexpidx,
  datafile = fullfile(expdirs{expi},'data.mat');
  load(datafile,'y')
  ally{expi} = y;
end

desiredexpfracs = [.6,.2,.1,.1];
desiredlabelfracs = [0,.4,.2,.2];
splitnames = {'usertrain','testtrain','test1','test2'};

yidxmanual = find(~cellfun(@isempty,regexp(ynames,'^perframe_','once')));
nboutsmanualperexp = zeros([nexps,numel(yidxmanual),2]);
nframesmanualperexp = zeros([nexps,numel(yidxmanual),2]);
for k = 1:numel(yidxmanual),
  j = yidxmanual(k);
  for expi = finalexpidx,    
    ally{expi}(:,:,j);
    nflies = size(ally{expi},1);
    for fly = 1:nflies,
      for v = [0,1],
        isv = ally{expi}(fly,:,j)==v;
        t0s = get_interval_ends(isv);
        nboutsmanualperexp(expi,k,v+1) = nboutsmanualperexp(expi,k,v+1) + numel(t0s);
        nframesmanualperexp(expi,k,v+1) = nframesmanualperexp(expi,k,v+1) + nnz(isv);
      end
    end
  end
  
end


% reseed the random number generator so that this code always produces the
% same results
rng('default');

% for each label type, select videos to be train, test1, test2 
exp2split = nan(1,nexps);
costpertype = nan(1,numel(labels_include));
assignstatspertype = [];
for k = 1:numel(labels_include),
  
  labelcurr = labels_include{k};
  expidxcurr = find(strcmp(labelcurr,{expinfo.label}));
  expidxcurr = intersect(expidxcurr,finalexpidx);
  nexpscurr = numel(expidxcurr);
  
  nboutsperexpcurr = nboutsmanualperexp(expidxcurr,:,:);
  [exp2splitcurr,costcurr,assignstatscurr] = AssignSplits(nboutsperexpcurr,desiredexpfracs,desiredlabelfracs);
  exp2split(expidxcurr) = exp2splitcurr;
  costpertype(k) = costcurr;
  if k == 1,
    assignstatspertype = repmat(assignstatscurr,[1,numel(labels_include)]);
  else
    assignstatspertype(k) = assignstatscurr; %#ok<SAGROW>
  end
  
end

% print stats about assignments chosen
nsplits = numel(splitnames);
fprintf('Number of experiments per split:\n');
for i = 1:nsplits,
  fprintf('  %s: %d (%f fraction)\n',splitnames{i},nnz(exp2split==i),nnz(exp2split==i)/numel(finalexpidx));
end
% Number of experiments per split:
%   usertrain: 43 (0.438776 fraction)
%   testtrain: 22 (0.224490 fraction)
%   test1: 15 (0.153061 fraction)
%   test2: 18 (0.183673 fraction)


splityframecount = cell(1,nclasses);
for k = 1:nclasses,
  splityframecount{k} = zeros(nsplits,numel(stats.y.uniquevals{k}));
end
for spliti = 1:nsplits,
  expidx = find(exp2split == spliti);
  for expi = expidx,
    y = ally{expi};
    yreshape = reshape(y,size(y,1)*size(y,2),size(y,3));
    for k = 1:nclasses,
      if all(isnan(yreshape(:,k))),
        continue;
      end
      c = hist(yreshape(~isnan(yreshape(:,k)),k),stats.y.uniquevals{k});
      splityframecount{k}(spliti,:) = splityframecount{k}(spliti,:) + c;
    end
  end
end

hfig = 3;
figure(hfig);
c = ceil(sqrt(nclasses));
r = ceil(nclasses/c);
hax = createsubplots(r,c,[.025,.05]);
for k = 1:nclasses,
  
  h = bar(hax(k),splityframecount{k}',1);
  title(hax(k),ynames{k},'Interpreter','none');
  s = arrayfun(@(x) sprintf('%.2g',x),stats.y.uniquevals{k},'Uni',0);
  set(hax(k),'XTick',1:numel(stats.y.uniquevals{k}),'XTickLabel',s);
  set(hax(k),'box','off');
  [r1,c1] = ind2sub([r,c],k);
  if c1 == 1,
    ylabel(hax(k),'N. frames');
  end
  if r1 == r,
    xlabel(hax(k),'Value');
  end
  axisalmosttight([],hax(k));

  if k == 1,
    legend(h,splitnames);
  end
  
end

delete(hax(nclasses+1:end));
save2pdf('SplitLabelInfo.pdf',hfig);
save2png('SplitLabelInfo.png',hfig);

fprintf('Number of bouts per label, split (');
fprintf('%s ',splitnames{:});
fprintf(')\n');
fprintf('%25s |  ',' ');
for labeli = 1:nmanuallabels,
  for val = 0:1,
    fprintf(sprintf('%%%ds %%d|',nsplits*4),manualbehaviors{labeli},val);
  end
end
fprintf('\n');
for k = 1:numel(labels_include),
  idxcurr = strcmp(labels_include{k},{expinfo.label});
  fprintf('%25s ',labels_include{k});
  fprintf('|  ');
  for labeli = 1:nmanuallabels,
    for vali = 1:2,
      for spliti = 1:nsplits,
        fprintf('%3d ',sum(nboutsmanualperexp(exp2split==spliti&idxcurr,labeli,vali)));
      end
      fprintf('  |');
    end
  end
  fprintf('\n');
end

% Number of bouts per label, split (usertrain testtrain test1 test2 )
%                           |        aggression 0|      aggression 1|           chase 0|           chase 1|       courtship 0|       courtship 1|      highfence2 0|      highfence2 1|         wingext 0|         wingext 1|       wingflick 0|       wingflick 1|
%                   control |    0   2   0   5   |  0   0   0   0   |  0  12  16   8   |  0   0   0   0   |  6   9   7   2   |  0   0   0   0   |  5   5   4   4   |  0   1   0   0   |  0   5  10  29   |  0   0   0   1   | 10   3   0   3   |  0   1   0   0   |
%          BDP_sexseparated |    0   0   2   0   |  0   0   1   0   |  0  46   9  15   |  0  17   6   5   |  3   4   0   3   |  9   2   0   0   |  0   0   0   0   |  0   0   0   0   |  0  18  35  28   |  0  11   9  14   | 13   8   4   6   | 15  21   0   5   |
%               Control_RGB |    0   4   0   2   |  0   3   0   2   |  0  13  12  21   |  0   2   0   2   |  2   3   4   3   |  0   0   0   0   |  3   2   0   3   |  0   0   0   2   |  0  25  20  21   |  0   0   0   0   |  0   6   6   4   |  0   3   7   5   |
%                     71G01 |    0   3   0   3   |  0   0   0   2   |  0  57  40  39   |  0  33  15  16   |  0  10   3   3   |  0   7   3   2   |  2   4   2   7   |  2   3   0   2   |  0  29  39  59   |  0  13  23  28   | 12   5   5  12   | 23  12  13  14   |
%       male71G01_femaleBDP |    0   0   0   3   |  0   0   0   0   | 44  45   0  30   | 25  23   0  30   |  6   3   2   0   |  5   4   3   0   |  0   0   6   0   |  0   0   1   0   | 38  29   0  24   | 26  14   0  12   |  0   8   0  11   |  0   9   0  15   |
%                     65F12 |    0   2   3   0   |  0   0   0   0   | 14  32  37  50   | 10  17  18  32   |  3   4   3   4   |  5   3   1   1   |  8   6   1   7   |  0   0   0   1   |  0   6  25  12   |  0   7  10   8   | 18   9   5   5   | 27  14   6   2   |
%                     91B01 |    0   0   4   2   |  0   0   0   0   |  0  45  29  18   |  0   0   0   0   | 19   5   6   0   |  0   0   0   0   |  2   2   4   4   |  0   0   0   0   |  0  12   6   8   |  0   0   0   0   | 20  14   0   9   |  0   0   0   0   |
%              BlindControl |    1   2   0   2   |  2   0   0   2   |  0  33  35  27   |  0   2   0   0   |  3   4   4   4   |  0   0   0   0   | 16   4   5   0   |  4   1   2   0   |  0  30   5  11   |  0   0   0   0   |  7   9   0   5   |  2   6   0   5   |
%    aIPgpublished3_newstim |    9  10  10  10   |  8  10  10  10   |  0  21  13  11   |  0   8   3   3   | 18   6   4   6   |  0   0   0   0   |  7   4   5   7   | 18  10   6   9   |  0  22  14  12   |  0   0   0   0   |  5   7   0  12   | 11   3   0  20   |
%    pC1dpublished1_newstim |    0  11  11  11   |  0  11  10  11   |  0  20  11  20   |  0   3   2  12   |  0   4   5   0   |  0   0   0   0   |  6   5   1   3   |  9   9   7  12   |  0  19  16   9   |  0   0   0   0   |  6   6  10   6   |  8   9  12   5   |
%         aIPgBlind_newstim |    5   2   3   6   |  5   3   3   2   |  0  26  21  29   |  0   0   0   0   |  6   5   4   5   |  0   0   0   0   |  0  10   4   6   |  0   4   4   4   |  0  23  11  14   |  0   0   0   0   |  4   6   7   6   | 15   0   4   0   |

fid = fopen('SplitInfo.csv','w');
fprintf(fid,'Experiment,SplitName,SplitIdx\n');
for expi = finalexpidx,
  expname = fileBaseName(expdirs{expi});
  fprintf(fid,'%s,%s,%d\n',expname,splitnames{exp2split(expi)},exp2split(expi));
end
fclose(fid);

%% make apt results videos

override_datalocparams = struct;
override_datalocparams.apttrkfilestr = 'apttrk.mat';
override_datalocparams.aptresultsavefilestr = 'apt_results_movie';
override_ctraxparams = struct;
override_ctraxparams.figpos = [1,1,1200,856];
override_ctraxparams.nframes = [1500 1500 1500];

override_ctraxparams_opto = override_ctraxparams;
override_ctraxparams_opto.nframes = 1000;
override_ctraxparams_opto.nframes_beforeindicator = 500;

hidemovietype = true;
resultsvideo_nintervals = 3;

save OverrideParams_MakeAPTResultsMovie.mat override_datalocparams override_ctraxparams override_ctraxparams_opto hidemovietype resultsvideo_nintervals

mintimestamp = datenum('20211224T154800','yyyymmddTHHMMSS');
%for expii = 1:nexps,
for expi = 1:nexps,
  %expi = exporder(expii);
  
  if ~isfinaldata(expi) || isbad(expi),
    continue;
  end
  
  expdir = expdirs{expi};
  [~,expname] = fileparts(expdir);
  fprintf('Exp %d/%d %s, %s\n',expi,nexps,expname,expinfo(expi).label);
  
  istrp = ~isempty(regexp(expname,'nochr_Trp','once'));
  
  if istrp,
    analysis_protocol_curr = reg_trp_analysis_protocol;
    override_ctraxparams_varcurr = 'override_ctraxparams';
  else
    analysis_protocol_curr = analysis_protocol;
    override_ctraxparams_varcurr = 'override_ctraxparams_opto';
  end
  
  logfile = fullfile(rundir,sprintf('aptresmov_%s_%s.out',expname,timestamp));
  resfile = fullfile(rundir,sprintf('apt_%s_%s_bsub.out',expname,timestamp));
  reserrfile = fullfile(rundir,sprintf('apt_%s_%s_bsub.err',expname,timestamp));
  
  movfile = fullfile(expdir,sprintf('%s_%s.mp4',override_datalocparams.aptresultsavefilestr,expname));
  if exist(movfile,'file')
    tmp = dir(movfile);
    if tmp.datenum >= mintimestamp,
      fprintf('APT results video exists for %d, %s, skipping\n',expi,expname);
      continue;
    else
      delete(movfile);
    end
  end
  
  % FlyDiscoMakeAPTResultsMovie(expdir,'settingsdir',reg_settingsdir,'analysis_protocol',analysis_protocol_curr,'override_datalocparams',override_datalocparams,'override_ctraxparams',eval(override_ctraxparams_varcurr),'hidemovietype',hidemovietype,'nintervals',resultsvideo_nintervals);
  matlabcmd = sprintf('addpath %s; APT.setpath; addpath %s; modpath; load(''%s/OverrideParams_MakeAPTResultsMovie.mat''); FlyDiscoMakeAPTResultsMovie(''%s'',''settingsdir'',''%s'',''analysis_protocol'',''%s'',''override_datalocparams'',override_datalocparams,''override_ctraxparams'',%s,''hidemovietype'',hidemovietype,''nintervals'',resultsvideo_nintervals);',aptpath,fdapath,cwd,expdir,reg_settingsdir,analysis_protocol_curr,override_ctraxparams_varcurr);
  jobcmd = sprintf('cd %s; %s -nodisplay -batch "%s exit;" > %s 2>&1',cwd,matlabpath,matlabcmd,logfile);
%   sprintf('ssh login1 "bsub -n2 -J %s -o %s -e %s \\\"cd %s; %s -nodisplay -batch \\\\\\"addpath %s; modpath; FlyDiscoClassifySex(''%s'',''settingsdir'',''%s'',''analysis_protocol'',''%s''); exit;\\\\\\" > %s 2>&1\\\""',...
%     expname,resfile,resfile,cwd,matlabpath,fdapath,expdir,reg_settingsdir,analysis_protocol,logfile);
  bsubcmd = sprintf('bsub -n 2 -J aptmov_%s -o %s -e %s "%s"',expname,resfile,reserrfile,strrep(jobcmd,'"','\"'));
  sshcmd = sprintf('ssh login1 "%s"',strrep(strrep(bsubcmd,'\','\\'),'"','\"'));
  disp(sshcmd);
  unix(sshcmd);
  
end

%% 

resultsmovieisdone = false(1,nexps);
for expi = 1:nexps,
  
  if ~isfinaldata(expi) || isbad(expi),
    continue;
  end
  
  expdir = expdirs{expi};
  [~,expname] = fileparts(expdir);
  movfile = fullfile(expdir,sprintf('%s_%s.mp4',override_datalocparams.aptresultsavefilestr,expname));
  
  if exist(movfile,'file')
    tmp = dir(movfile);
    if tmp.datenum >= mintimestamp,
      resultsmovieisdone(expi) = true;
    end
  end  
end
  
%%
% 
% isfinaldata = false(1,nexps);
% for expi = 1:nexps,
%   isfinaldata(expi) = exist(fullfile(expdirs{expi},'data.mat'),'file')>0;
% end

%%
% 
% for i = find(~isclassifiedsex),
%   if ~exist(fullfile(expdirs{i},'registered_trx.mat'),'file'),
%     unix(sprintf('ln -s %s/registered_trx.mat %s/registered_trx.mat',expinfo(i).file_system_path,expdirs{i}));
%   end
% end

%% check for errors

for expi = 1:nexps,
  %expi = exporder(expii);
  
  if ~isfinaldata(expi) || isbad(expi),
    continue;
  end
  
  expdir = expdirs{expi};
  [~,expname] = fileparts(expdir);
  fprintf('Exp %d/%d %s, %s\n',expi,nexps,expname,expinfo(expi).label);
  
  istrp = ~isempty(regexp(expname,'nochr_Trp','once'));
  
  if istrp,
    analysis_protocol_curr = reg_trp_analysis_protocol;
  else
    analysis_protocol_curr = analysis_protocol;
  end
  
  logfile = fullfile(rundir,sprintf('aptresmov_%s_%s.out',expname,timestamp));
  resfile = fullfile(rundir,sprintf('apt_%s_%s_bsub.out',expname,timestamp));
  reserrfile = fullfile(rundir,sprintf('apt_%s_%s_bsub.err',expname,timestamp));
  
  movfile = fullfile(expdir,sprintf('%s_%s.mp4',override_datalocparams.aptresultsavefilestr,expname));
  if exist(movfile,'file')
    tmp = dir(movfile);
    if tmp.datenum >= mintimestamp,
      fprintf('APT results video exists for %d, %s, skipping\n',expi,expname);
      continue;
    end
  end
  
  if exist(reserrfile,'file'),
    fprintf('Err file:\n');
    type(reserrfile);
  end
  if exist(resfile,'file'),
    fprintf('Res file:\n');
    type(resfile);
  end
  if exist(logfile,'file'),
    type(logfile);
  end
  
  input(sprintf('%d: %s: ',expi,expname));
  
end

%% compute per-frame features necessary for JAABA

[featureLexicon,animalType] = featureLexiconFromFeatureLexiconName('flies_disco',fullfile(fdapath,'JAABA'));
perframefns = fieldnames(featureLexicon.perframe);
forcecompute = true;

save OverrideParams_ComputePerFrameFeatures.mat perframefns forcecompute;

for expi = 1:nexps,
  %expi = exporder(expii);
  
  if ~isfinaldata(expi) || isbad(expi),
    continue;
  end
  
  expdir = expdirs{expi};
  [~,expname] = fileparts(expdir);
  fprintf('Exp %d/%d %s, %s\n',expi,nexps,expname,expinfo(expi).label);
  
  logfile = fullfile(rundir,sprintf('pffmov_%s_%s.out',expname,timestamp));
  resfile = fullfile(rundir,sprintf('pff_%s_%s_bsub.out',expname,timestamp));
  reserrfile = fullfile(rundir,sprintf('pff_%s_%s_bsub.err',expname,timestamp));

  pffmat = dir(fullfile(expdir,'perframe','*.mat'));
  pffmat([pffmat.datenum] < mintimestamp) = [];
  pff = cell(size(pffmat));
  for j = 1:numel(pffmat),
    [~,pff{j}] = fileparts(pffmat(j).name);
  end
  todo = setdiff(perframefns,pff);
  if isempty(todo),
    fprintf('Per-frame features exist for %d, %s, skipping\n',expi,expname);
    continue;
  end
  
  matlabcmd = sprintf('addpath %s; APT.setpath; addpath %s; modpath; load(''%s/OverrideParams_ComputePerFrameFeatures.mat''); FlyDiscoComputePerFrameFeatures(''%s'',''settingsdir'',''%s'',''analysis_protocol'',''%s'',''perframefns'',perframefns,''forcecompute'',forcecompute);',aptpath,fdapath,cwd,expdir,settingsdir,analysis_protocol);
  jobcmd = sprintf('cd %s; %s -nodisplay -batch "%s exit;" > %s 2>&1',cwd,matlabpath,matlabcmd,logfile);
  bsubcmd = sprintf('bsub -n 2 -J pff_%s -o %s -e %s "%s"',expname,resfile,reserrfile,strrep(jobcmd,'"','\"'));
  sshcmd = sprintf('ssh login1 "%s"',strrep(strrep(bsubcmd,'\','\\'),'"','\"'));
  disp(sshcmd);
  unix(sshcmd);
  
end

%% check pff

ispff = false(1,nexps);
for expi = 1:nexps,
  %expi = exporder(expii);
  
  if ~isfinaldata(expi) || isbad(expi),
    continue;
  end
  
  expdir = expdirs{expi};
  
  pffmat = dir(fullfile(expdir,'perframe','*.mat'));
  pffmat([pffmat.datenum] < mintimestamp) = [];
  pff = cell(size(pffmat));
  for j = 1:numel(pffmat),
    [~,pff{j}] = fileparts(pffmat(j).name);
  end
  todo = setdiff(perframefns,pff);
  ispff(expi) = isempty(todo);
  
end

%%

fid = fopen('MABE_explist20211102.txt','w');
fprintf(fid,'%s\n',expdirs{isfinaldata & ~isbad});
fclose(fid);

%% print final stats

idxgood = find(isfinaldata & ~isbad);

% how many videos per condition
[labels,~,labelidx] = unique({expinfo(idxgood).label});
counts = hist(labelidx,1:numel(labels));
for i = 1:numel(labels),
  fprintf('%s: %d\n',labels{i},counts(i));
end

% how much time for each condition? 
frspercondition = zeros(1,numel(labels));
fliespercondition = zeros(1,numel(labels));
for i = 1:numel(idxgood),
  expdir = expdirs{idxgood(i)};
  [~,nframes,fid] = get_readframe_fcn(fullfile(expdir,'movie.ufmf'));
  if fid > 1, fclose(fid); end
  nflies = nmales(i) + nfemales(i);
  frspercondition(labelidx(i)) = frspercondition(labelidx(i)) + nframes;
  fliespercondition(labelidx(i)) = fliespercondition(labelidx(i)) + nflies;
end
fps = 150;
for i = 1:numel(labels),
  fprintf('%s: %d videos, %d flies, %f minutes\n',labels{i},counts(i),fliespercondition(i),frspercondition(i)/fps/50);
end
for i = 1:numel(labels),
  fprintf('%s:\n',labels{i});
  for j = find(labelidx==i),
    fprintf('%s\n',expdirs{idxgood(j)});
  end
end

% 65F12: 12 videos, 116 flies, 72.013467 minutes
% 71G01: 13 videos, 125 flies, 84.014400 minutes
% 91B01: 10 videos, 98 flies, 60.011200 minutes
% BDP_sexseparated: 4 videos, 38 flies, 24.004667 minutes
% BlindControl: 9 videos, 77 flies, 70.950933 minutes
% Control_RGB: 6 videos, 55 flies, 47.303200 minutes
% aIPgBlind_newstim: 11 videos, 88 flies, 86.679333 minutes
% aIPgpublished3_newstim: 9 videos, 85 flies, 70.966800 minutes
% control: 9 videos, 83 flies, 54.010000 minutes
% male71G01_femaleBDP: 5 videos, 48 flies, 30.007067 minutes
% pC1dpublished1_newstim: 8 videos, 75 flies, 63.072133 minutes
% 65F12:
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigA_20201212T163531
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigA_20201216T155516
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigA_20201216T175643
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigB_20201212T163629
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigB_20201216T155701
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigB_20201216T175725
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigC_20201212T163727
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigC_20201216T155818
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigC_20201216T175810
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigD_20201212T163812
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigD_20201216T155952
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA65F12_Unknown_RigD_20201216T175902
% 71G01:
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigA_20201212T162201
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigA_20201216T153505
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigA_20201216T182330
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigB_20201212T162251
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigB_20201216T153635
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigB_20201216T182433
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigC_20201212T162347
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigC_20201216T153727
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigC_20201216T182515
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigD_20201209T164717
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigD_20201212T162439
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigD_20201216T153831
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigD_20201216T182616
% 91B01:
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA91B01_Unknown_RigA_20201212T170356
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA91B01_Unknown_RigA_20201216T152414
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA91B01_Unknown_RigB_20201212T170551
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA91B01_Unknown_RigB_20201216T152556
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA91B01_Unknown_RigB_20201216T183652
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA91B01_Unknown_RigC_20201212T170550
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA91B01_Unknown_RigC_20201216T152707
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA91B01_Unknown_RigC_20201216T183744
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA91B01_Unknown_RigD_20201212T170632
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA91B01_Unknown_RigD_20201216T183848
% BDP_sexseparated:
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpApBDP_Unknown_RigA_20201216T170209
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpApBDP_Unknown_RigB_20201216T170433
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpApBDP_Unknown_RigC_20201216T170601
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpApBDP_Unknown_RigD_20201216T170724
% BlindControl:
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA5_JHS_K_85321_RigA_20210923T071217
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA5_JHS_K_85321_RigA_20210923T085131
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA5_JHS_K_85321_RigD_20210923T082843
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA_JHS_K_85321_RigA_20210903T065907
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA_JHS_K_85321_RigB_20210903T065743
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA_JHS_K_85321_RigB_20210903T074659
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA_JHS_K_85321_RigB_20210903T085102
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA_JHS_K_85321_RigC_20210902T064801
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA_JHS_K_85321_RigC_20210903T065147
% Control_RGB:
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr5_JHS_K_85321_RigA_20210923T082323
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr5_JHS_K_85321_RigC_20210923T070407
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JHS_K_85321_RigA_20210902T074347
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JHS_K_85321_RigA_20210903T085205
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JHS_K_85321_RigC_20210903T060158
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JHS_K_85321_RigD_20210903T075652
% aIPgBlind_newstim:
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA_JRC_SS36564_RigA_20210902T072746
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA_JRC_SS36564_RigA_20210903T081259
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA_JRC_SS36564_RigB_20210902T072658
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA_JRC_SS36564_RigB_20210903T081150
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA_JRC_SS36564_RigC_20210903T061318
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA_JRC_SS36564_RigD_20210903T072648
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA_JRC_SS36564_RigD_20210903T084648
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA5_JRC_SS36564_RigA_20210923T075505
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA5_JRC_SS36564_RigA_20210923T083705
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA5_JRC_SS36564_RigB_20210923T070227
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/NorpA5_JRC_SS36564_RigB_20210923T085311
% aIPgpublished3_newstim:
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JRC_SS36564_RigA_20210903T072524
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JRC_SS36564_RigB_20210903T072353
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JRC_SS36564_RigC_20210902T061726
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JRC_SS36564_RigC_20210903T071640
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JRC_SS36564_RigC_20210903T074158
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JRC_SS36564_RigD_20210902T072259
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JRC_SS36564_RigD_20210903T080906
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr5_JRC_SS36564_RigA_20210923T070124
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr5_JRC_SS36564_RigB_20210923T074202
% control:
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpApBDP_Unknown_RigA_20201212T164930
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpApBDP_Unknown_RigA_20201216T174421
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpApBDP_Unknown_RigB_20201212T165050
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpApBDP_Unknown_RigB_20201216T160731
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpApBDP_Unknown_RigB_20201216T174420
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpApBDP_Unknown_RigC_20201212T165139
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpApBDP_Unknown_RigC_20201216T160840
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpApBDP_Unknown_RigC_20201216T174511
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpApBDP_Unknown_RigD_20201216T161008
% male71G01_femaleBDP:
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigA_20201212T171758
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigA_20201216T162938
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigB_20201212T172153
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigB_20201216T163123
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/nochr_TrpA71G01_Unknown_RigC_20201216T163316
% pC1dpublished1_newstim:
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JRC_SS56987_RigA_20210902T070106
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JRC_SS56987_RigA_20210902T075834
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JRC_SS56987_RigB_20210902T065953
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JRC_SS56987_RigC_20210903T064025
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr_JRC_SS56987_RigD_20210903T083342
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr5_JRC_SS56987_RigC_20210923T074356
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr5_JRC_SS56987_RigC_20210923T082707
% /groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data/CsChr5_JRC_SS56987_RigD_20210923T070522


%% look at the background image for each video

isokbg = nan(1,nexps);

for expi = 1:nexps,
  %expi = exporder(expii);
  
  if ~isfinaldata(expi) || isbad(expi),
    continue;
  end

  expdir = expdirs{expi};
  [~,expname] = fileparts(expdir);
  
  bd = load(fullfile(expdir,'movie-bg.mat'));
  clf;
  imagesc(bd.bg.bg_mean,[0,1]);
  axis image;
  isokbg(expi) = input(sprintf('%d: %s: ',expi,expname));
end

%% look for changes in sex classification

for expi = 1:nexps,
  %expi = exporder(expii);
  
  if ~isfinaldata(expi) || isbad(expi),
    continue;
  end

  expdir = expdirs{expi};
  [~,expname] = fileparts(expdir);

  load(fullfile(expdir,'data.mat'),'y');
  
  isswitch = ~isnan(y(:,1:end-1,yidx.female)) & ~isnan(y(:,2:end,yidx.female)) & ...
    y(:,1:end-1,yidx.female)~=y(:,2:end,yidx.female);
  [frs,flies] = find(isswitch');
  
  if ~isempty(flies),
    fprintf('%d %s:\n',expi,expname);
    fprintf('Fly\tFrame\n');
    for j = 1:numel(frs),
      fprintf('%d\t%d\n',flies(j),frs(j));
    end
  end
  
end

%%

if ~exist(finaldatadir,'dir'),
  mkdir(finaldatadir);
end

tocopy = {'apt_results_movie_<expname>.mp4','data.mat'};
destname = {'preview.mp4','data.mat'};

for expi = 1:nexps,
  if ~isfinaldata(expi) || isbad(expi),
    continue;
  end
  expdir = expdirs{expi};
  [~,expname] = fileparts(expdir);
  finalexpdir = fullfile(finaldatadir,expname);
  if ~exist(finalexpdir,'dir'),
    mkdir(finalexpdir);
  end
  for i = 1:numel(tocopy),
    filestr = strrep(tocopy{i},'<expname>',expname);
    destfilestr = strrep(destname{i},'<expname>',expname);
    sourcefile = fullfile(expdir,filestr);
    destfile = fullfile(finalexpdir,destfilestr);
    if exist(destfile,'file'),
      continue;
    end
    cmd = sprintf('ln -s %s %s',sourcefile,destfile);
    unix(cmd);
  end
  
end