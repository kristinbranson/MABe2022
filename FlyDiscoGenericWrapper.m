function FlyDiscoGenericWrapper(fun,matfile)

load(matfile); %#ok<LOAD>
feval(fun,expdir,settingsdir,analysis_protocol,dataloc_params,forcecompute);