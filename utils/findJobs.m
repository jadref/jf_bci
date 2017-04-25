function [collected]=collect_res(jobs,labstrs)
if ( nargin < 1 | isempty(jobs) ) jobs=sgejobs_pending; end;
if ( nargin < 2 ) labstrs=[]; end;

fprintf('Collecting the results from the cluster\n');
fnames=cell(0);labs=cell(0);collected=[];
for i=1:numel(jobs);
   if ( ~finished(jobs(i)) )
      fprintf('Not finished: %d) %s\n',i,jobs(i).description);
      continue;
   end
   if ( ~ok(jobs(i)) )
      fprintf('Not OK      : %d) %s\n',i,jobs(i).description);
      continue;
   end
   if ( ~isempty(labstrs) && isempty(regexp(jobs(i).description,labstrs)))
      fprintf('Not matched : %d) %s\n',i,jobs(i).description);
      continue;
   end;
   fprintf('Collecting  : %d) %s\n',i,jobs(i).description);
   collected=[collected;i];
end
