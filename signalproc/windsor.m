function [X]=windsor(X,dim,mode,thresh,lentype,verb);
% squash the data to reduce the effect of outliers
%
%   [X]=windsor(X,dim,mode,thresh,lentype,verb);
%
% Inputs:
%   X - [n-d] data
%   dim - [int] dimension of X to windsorize
%   mode - 'str' type of data squashing to do.   ('huber')
%          one of: 'median','huber','sqrt'
%   thresh - [float] threshold for labeling as outlier.  outlier if x > thresh*lenghtScale
%   lentype - 'str' method used to compute the lengthScale
%             one of : 'rms','mad'
if ( nargin<2 || isempty(dim) )    dim=-1; end
if ( nargin<3 || isempty(mode) )   mode='huber'; end;
if ( nargin<4 || isempty(thresh) ) thresh=3; end;
if ( nargin<5 || isempty(lentype) )lentype='std'; end;
if ( nargin<6 || isempty(verb) )   verb=1; end;
MAXEL=1e6;
dim(dim<0)=ndims(X)+dim(dim<0)+1;
% process the data in pieces
[idx,chkStrides,nchnks]=nextChunk([],size(X),dim,MAXEL);
ci=0; if ( verb >= 0 && nchnks>1 ) fprintf('%s:',mfilename); end;
while ( ~isempty(idx) )
  tmp = X(idx{:});
  
  med =median(tmp,dim);
  cX  =repop(tmp,'-',med); % distance from center
  
  if ( any(strcmp(lentype,{'std','rms'})) ) % std w.r.t cent	 
	 feat    =cX.^2;
  elseif ( strcmp(lentype,'mad') ) % median absolute distance
	 feat    =abs(cX);
  end

  isoutlier=false(size(tmp));
  for iter=1:3;
	 if ( any(strcmp(lentype,{'std','rms'})) ) % std w.r.t cent	 
		lenScale=sqrt(sum(feat,dim)./sum(~isoutlier,dim));
	 elseif ( strcmp(lentype,'mad') ) % median absolute distance
		lenScale=median(feat,dim);
	 end
	 outp =repop(cX,'>',lenScale*thresh); % outliers w.r.t cent+variance, too big
	 outn =repop(cX,'<',-lenScale*thresh); % outliers, too small
	 isoutlier = isoutlier | outp | outn;
	 feat(outp|outn)=0; % zero out these feature scores so approx ignored in length-scale-computation
  end
  
  outp =repop(cX,'>',lenScale*thresh); % outliers w.r.t cent+variance, too big
  outn =repop(cX,'<',-lenScale*thresh); % outliers, too small
  % fix/update the outliers. move outliers to the outer data limits
  if ( strcmp(mode,'median') ) % replace with mean
	 tt=repmat(med,[ones(1,dim-1) size(tmp,dim) ones(1,ndims(tmp)-(dim+1))]);
	 tmp(outp | outn)=tt(outp|outn);
	 X(idx{:})=tmp;

  elseif ( strcmp(mode,'huber') ) % replace with mean+thresh*std
    % BODGE: this is horrible as we do lots of extra work changing every entry which is uncessary....
	 if ( any(outp(:)) )
		ub = med + lenScale*thresh;
		tt = repmat(ub,[ones(1,dim-1) size(tmp,dim) ones(1,ndims(tmp)-(dim+1))]);
		tmp(outp) = tt(outp);
	 end

	 if ( any(outn(:)) )
		lb = med - lenScale*thresh;
		tt = repmat(lb,[ones(1,dim-1) size(tmp,dim) ones(1,ndims(tmp)-(dim+1))]);
		tmp(outn) = tt(outn);
	 end
	 X(idx{:}) = tmp;

  elseif ( strcmp(mode,'sqrt') ) % sqrt with mean+thresh*std as point of no change
	 % BODGE: this is horrible as we do lots of extra work changing every entry which is uncessary....
	 cX = sqrt(abs(cX)); % sqrt the distance
	 cX = repop(cX,'*',sqrt(lenScale*thresh)); % re-scale to thresh*std as point of no change 
	 tt = repop(med,'+',cX);          tmp(outp) = tt(outp);
	 tt = repop(med,'-',cX);          tmp(outn) = tt(outn);
	 X(idx{:})=tmp;
  end  

  if ( verb >=0 && nchnks>1 ) ci=ci+1; textprogressbar(ci,nchnks);	end
  idx=nextChunk(idx,size(X),chkStrides);
end
return;
%-----------------------------------------
function testcase()
% simple gaussian outlier problem
nd=10; nSamp=100; nOut=10;
X=randn(nd,nSamp);
X(:,1:nOut)=X(:,1:nOut) + randn(nd,nOut)*10; % add outliers
  
Xw=windsor(X,[],'huber');
Xw=windsor(X,[],'sqrt');
Xw=windsor(X,[],'sqrt',[],'mad');
clf;mcplot(Xw','linewidth',2)
clf;mimage(X,Xw,'diff',1)
