function res=rmPrepInfo(res,rmX,verb)
% remove extra prep info and data from data structures which only contain results
if ( nargin<2 || isempty(rmX) ) rmX=true; end;
if ( nargin<3 || isempty(verb) ) verb=0; end;
if ( verb>0) fprintf('rmPrepInfo:'); end;
for ri=1:numel(res);
   if ( iscell(res) ) resi=res{ri}; else resi=res(ri); end;
   if ( numel(resi)>1 || iscell(resi) )  % recurse for array inputs
     resi=rmPrepInfo(resi); 
   else
     if ( isfield(resi,'prep') )
       for pi=1:numel(resi.prep)-1;
         resi.prep(pi).info=[]; 
       end
       if ( isfield(resi.prep(end).info,'res') )
         if ( isfield(resi.prep(end).info.res,'fold') ) % clean up the fold info
           resi.prep(end).info.res.fold.f=[]; 
           resi.prep(end).info.res.fold.soln=[];
           resi.prep(end).info.res.fold.di=[];
         end
         if ( isfield(resi.prep(end).info.res,'outer') ) % clean up the outer soln info
           resi.prep(end).info.res.outer=[];
         end
         if ( isfield(resi.prep(end).info.res,'inner') ) % clean up the outer soln info
           resi.prep(end).info.res.inner=[];
         end
         if ( isfield(resi.prep(end).info.res,'opt') )
            if ( isfield(resi.prep(end).info.res.opt,'zopt') )
               resi.prep(end).info.res.opt.zopt=[];
            end
         end
         %resi.prep(end).info.res.soln=[];
         resi.prep(end).info.res.f=[];
         resi.prep(end).info.res.tstf=[]; % keep fold summary info 
         resi.prep(end).info.odi  =[];
         if( rmX ) resi.X=[]; end% remove predictions as uncessary
       end
     end
     if( isfield(resi,'Ydi') ) 
        for yi=1:numel(resi.Ydi); resi.Ydi(yi).extra=[]; end;
     end;
     if( rmX && isfield(resi,'foldIdxs') ) resi.foldIdxs=[]; end;
     if( isfield(resi,'di') )  % strip large bits from the general dim-info
        for yi=1:numel(resi.di); resi.di(yi).extra=[];resi.di(yi).info=[]; end;
     end;
    end
   if ( iscell(res) ) res{ri}=resi; else res(ri)=resi; end;
   if ( verb>0 ) textprogressbar(ri,numel(res)); end;
end
