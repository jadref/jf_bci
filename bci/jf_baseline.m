function [z,opts]=jf_baseline(z,varargin)
% re-reference with a weighted mean over a subset of indicies
% Options:
%  dim -- dimension(s) to be base-lined ('time')
%  idx -- which elements along 'dim' to included in and re-referenced (1:size(z.X,dim))
%  period-- weighting for the included elements, average over idx if empty.
%         OR
%         [2/3/4x1] spec of which points to use for the base-line computation in format 
%                    as used for mkFilter, e.g. [-100 -400] means from -100ms->400ms
%  op  -- [str] operator to us for the referencing ('-')
%  summary -- additional summary string
opts=struct('period',[],'subIdx',[],'verb',0);
[opts,varargin]=parseOpts(opts,varargin);
z=jf_reref(z,'wght',opts.period,varargin);
z.prep(end).info.method=mfilename;
return;
%--------------------------------------------------------------------------
function testCase()
z=jf_load('external_data/mpi_tuebingen/vgrid/nips2007/1-rect230ms','jh','flip_opt');
z=jf_load('external_data/mlsp2010/p300-comp','s1','trn');

%z=jf_retain(z,'dim','time','idx',1:600);
zmu=jf_reref(z,'dim','ch','idx',[z.di(1).extra.iseeg]);
zrmu=jf_reref(z,'dim','ch','idx',[z.di(1).extra.iseeg],'wght','robust');
figure(1);clf;jf_plotEEG(z,'subIdx',{[] [] 1});    
figure(2);clf;jf_plotEEG(zmu,'subIdx',{[] [] 1});  saveaspdf('~/car');
figure(3);clf;jf_plotEEG(zrmu,'subIdx',{[] [] 1}); saveaspdf('~/rcar');

figure(1);clf;jf_plotERP(z,'subIdx',{[1:27 29:56 58:64] [] []});   saveaspdf('~/raw_ERP');
figure(2);clf;jf_plotERP(zmu,'subIdx',{[1:27 29:56 58:64] [] []}); saveaspdf('~/car_ERP');
figure(3);clf;jf_plotERP(zrmu,'subIdx',{[1:27 29:56 58:64] [] []});saveaspdf('~/rcar_ERP');
