function [z]=jf_dimseqPerf(z,varargin)
%  simple sequence performance comp where we just average away dim
  opts=struct('dim','window','lossFn','bal');
  opts=parseOpts(opts,varargin);
Y  =z.Y;
dim=opts.dim;
if ( isfield(z,'Ydi') ) dim=n2d(z.Ydi,dim); spD=n2d(z.Ydi,'subProb'); else error('Only if theres a Ydi'); end
Yd =max(z.Y,[],dim); % the dim label, ignoreing 0-labeled (i.e. ignored) entries
res=z.prep(end).info.res;
if ( isfield(res,'opt') && isfield(res.opt,'tstf') )
  dv=res.opt.tstf;
  dv=reshape(dv,size(Y)); % should match
  dvd=sum(dv,dim);
  res.rawtst = res.tst; res.rawtstconf=res.tstconf;
  res.tstconf= dv2conf(Yd,dvd,[setdiff(1:ndims(Yd),spD) spD]);
  res.tst    = conf2loss(res.tstconf,1,opts.lossFn);
  z.prep(end).info.res=res;
  fprintf('(seq) NA /%4.2f\n',res.tst);
else
  error('only when classifier stores the tstf');
end
return;
%--------------------------
function testCase()
  oz=jf_mksfToy();
  z=oz;
  %z=jf_retain(z,'dim','time','range','between','vals',trl_rng);
  z=jf_windowData(z,'dim','time','width_ms',1000); % cut into sub-trials
  z=jf_labeldim(z,'dim','window');
  z=jf_welchpsd(z,'width_ms',250);
  jf_disp(z);
  z=jf_cvtrain(z,'objFn','mlr_cg','binsp',0); % train clsfr on sub-trials
  z=jf_dimseqPerf(z,'dim','window'); % get back to whole trial performance
