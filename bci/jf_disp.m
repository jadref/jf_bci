function [summary]=jf_disp(z)
% generate a nice summary of the data and history of the jf-data-struct
if ( numel(z)>1 || iscell(z) )
   summary=[];
   for i=1:numel(z); 
      if ( iscell(z) ) si=jf_disp(z{i}); else si=jf_disp(z(i)); end; 
      summary=[summary sprintf('\n%d)',i) si]; 
   end;
   return;
end
if ( isfield(z,'expt') ) expt=z.expt; else expt='?'; end;
if ( isfield(z,'subj') ) subj=z.subj; else subj='?'; end;
if ( isfield(z,'label') ) label=z.label; else label='?'; end;
if ( isfield(z,'session') ) session=z.session; else session=''; end;
if ( isnumeric(session) ) session=sprintf('%d',session); end;
if ( isequal(expt,'?') && isequal(subj,'?') && isequal(label,'?') )
   summary='';
else
   summary=sprintf('%s \t %s \t %s \t (%s)\n',expt,subj,session,label);
end
if( isfield(z,'X') )
  szX=size(z.X);
  if ( numel(szX)==2 && szX(2)==1 ) szX(2)=[]; end; % fix 2d size issues, i.e. col should be 1d
  if( isfield(z,'di') )
    diok=true;     
    if ( numel(szX)<numel(z.di) ) szX(end+1:numel(z.di)-1)=1; end
    if ( numel(szX)>numel(z.di) ) diok=false;  end;
    for di=1:numel(z.di)-1; 
      if(numel(z.di(di).vals)~=szX(di) && size(z.di(di).vals,2)~=szX(di)) diok=false; break; end; 
    end;
      summary =[summary sprintf('%s',dispDimInfo(z.di))];
      if ( ~diok )
         warning('dimInfo doesnt match size of X');
         summary =sprintf('%s -> [%s%d] *BAD DI*',summary,sprintf('%d x ',szX(1:end-1)),szX(end));
      end
   else
      szX=size(z.X);
      summary =sprintf('%s[%s%d]',summary,sprintf('%d x ',szX(1:end-1)),szX(end));
   end
   if ( isnumeric(z.X) && ~isreal(z.X) ) 
      summary =[summary sprintf(' (%s complex)',class(z.X))];
   else
      summary =[summary sprintf(' (%s)',class(z.X))];
   end
elseif( isfield(z,'di') )
   summary =[summary sprintf('%s',dispDimInfo(z.di))];
end
if( isfield(z,'prep') ) 
   tmp={z.prep.timestamp;z.prep.method;z.prep.summary};
   tmp=[num2cell(1:numel(z.prep));tmp];
   summary=[summary sprintf('\n%2d) %s %14s - %s',tmp{:})];
end
summary=[summary sprintf('\n')];
if( isfield(z,'Ydi') && ~isempty(z.Ydi) )
   summary =[summary sprintf('Labels: %s',dispDimInfo(z.Ydi))];   
elseif ( isfield(z,'Y') && ~isempty(z.Y) )
   szY=size(z.Y);
   summary =[summary sprintf('Labels: [%s] *NO-DI*',[sprintf('%d',szY(1)) sprintf('x %d',szY(2:end))])];
end
if( isfield(z,'foldIdxs') && ~isempty(z.foldIdxs) )
  if( isfield(z,'outfIdxs') && ~isempty(z.outfIdxs) )
    summary =[summary sprintf('\t(%d/%d in/out-folds)',size(z.foldIdxs,ndims(z.foldIdxs)),size(z.outfIdxs,ndims(z.foldIdxs)))];
  else
    summary =[summary sprintf('\t(%d folds)',size(z.foldIdxs,ndims(z.foldIdxs)))];
  end
end
return;
