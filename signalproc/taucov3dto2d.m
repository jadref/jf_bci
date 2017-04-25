function cov=taucov3dto2d(cov,bias,taus,T)
% convert from [ch x ch x tau] to block diagonal matrix
% [ch x ch x tau] = [  [ch x ch x 1]     [ch x ch x 2]      [ch x ch x 3]  ... [ch x ch x tau]   ]
%                   [  [ch x ch x 2]'    [ch x ch x 1]      [ch x ch x 2]  ... [ch x ch x tau-1] ]
%                   [  [ch x ch x 3]'    [ch x ch x 2]'     [ch x ch x 1]  ... [ch x ch x tau-2] ]
%                   [       :                  :                 :         ...        :          ]
%                   [ [ch x ch x tau]' [ch x ch x tau-1]' [ch x ch x tau-2]' ... [ch x ch x 1]   ]
% (optional)        [ [bias x ch x tau]' [bias x ch x tau-1]' [bias x ch x tau-2]' ... [bias x ch x 1]]
if ( nargin<2 || isempty(bias) ) bias=0; end;
if ( nargin<3 || isempty(taus) ) taus=0:size(cov,3)-1; end;
if ( nargin<4 || isempty(T) )    T   =numel(taus)*10; end;
szcov=[size(cov) 1 1];
szBlk=szcov(1); 
if ( bias>0 ) szBlk=szBlk-1; end;
iis  =1:szBlk; % block indices
nBlk =szcov(3); % num blocks = num taus

tcov=cov; % copy the 3d version to fill in from
cov =zeros([szBlk*nBlk+(bias>0),szBlk*nBlk+(bias>0),prod(szcov(4:end))],class(tcov));
% fill in block diag structure
for tj=1:nBlk; % horz blocks
  for ti=1:tj-1; % vert blocks > above main diag
    taui=abs(ti-tj)+1; % what tau goes in this place
    cov((ti-1)*szBlk+(iis),(tj-1)*szBlk+(iis),:)=tcov(iis,:,taui,:);
  end;
  ti=tj; taui=1; 
  cov((tj-1)*szBlk+(iis),(tj-1)*szBlk+(iis),:)=squeeze(tcov(iis,:,taui,:)); % main diag
  for ti=tj+1:nBlk; % vert blocks, below main diag -> transposed
    taui=abs(ti-tj)+1; % what tau goes in this place
    for k=1:size(cov,3);
      cov((ti-1)*szBlk+(iis),(tj-1)*szBlk+(iis),k)=tcov(iis,:,taui,k)';
    end
  end
end
if ( bias>0 ) % add the bias row
	cov(end,1:end-1,:,:)=reshape(tcov(end,:,:,:),[prod(szcov(2:3)) szcov(4:end)]);
	cov(1:end-1,end,:,:)=reshape(tcov(end,:,:,:),[prod(szcov(2:3)) szcov(4:end)]);
	cov(end,end,:,:)    =bias; % bias is number of taus
end
cov=reshape(cov,[szBlk*nBlk+(bias>0),szBlk*nBlk+(bias>0),szcov(4:end)]);
return;
