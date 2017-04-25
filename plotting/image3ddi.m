function [varargout]=image3ddi(A,Adi,dim,varargin);
% Wrapper round image3d to make it work nicely with A + dimInfo
%
% [varargout]=image3ddi(A,Adi,dim,varargin);
if ( nargin < 3 ) dim=1; end;
if ( ischar(dim) ) dim=1; varargin={dim varargin{:}}; end;
nout=nargout;%nout = max(nargout,1)-1;
szA=size(A);
if ( numel(szA)>3 && any(szA==1) ) % squeeze to make 3d
  nrmD=numel(szA)-3; % num dims to remove if possible
  singD=find(szA==1); singD=singD(max(1,end-nrmD+1):end); % dim's to remove
  keepD=setdiff(1:numel(szA),singD);
  A  = reshape(A,szA(keepD));
  Adi= Adi([keepD(:); end]);
  dim= dim-sum(dim>singD); % shift dim back
end
if ( numel(Adi)<3 ) Adi(end+1:3)=mkDimInfo(1,1,'dummy');  end
if ( dim==1 ) xlab=''; else 
   xlab = Adi(1).name; 
   if( ~isempty(Adi(1).units) ) xlab=[xlab ' (' Adi(1).units ')']; end;
end;
ylab=Adi(2).name; 
if( ~isempty(Adi(2).units) ) ylab=[ylab ' (' Adi(2).units ')']; end;
zlab=Adi(3).name;
if( ~isempty(Adi(3).units) ) zlab=[zlab ' (' Adi(3).units ')']; end;
clab=Adi(end).name;
if( ~isempty(Adi(end).units) ) clab=[clab ' (' Adi(end).units ')']; end;
pos2d=[];
if ( isfield(Adi(dim).extra,'pos2d') ) 
   pos2d=[Adi(dim).extra.pos2d];
   if( size(unique(pos2d','rows'),1)<size(pos2d,2) || size(pos2d,2)<szA(dim)) pos2d=[]; end; % only if valid
end;
[varargout{1:nout}]=image3d(A,dim,'Xvals',Adi(1).vals,'Yvals',Adi(2).vals,'Zvals',Adi(3).vals,...
                            'Ydir','normal','xlabel',xlab,'ylabel',ylab,'zlabel',zlab,...
                            'plotPos',pos2d,'clabel',clab,varargin{:});
return;
%------------
function testCases();
