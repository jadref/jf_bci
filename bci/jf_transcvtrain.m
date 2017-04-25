function z=jf_transcvtrain(z,varargin)
% simple wrapper to setup a transfer-learning call by setting N2w=blockIdx
opts=struct('blockIdx','block','blockOptName','N2w','objFn','translinfnlr_cg');
[opts,varargin]=parseOpts(opts,varargin);
blockIdx=getBlockIdx(z,opts.blockIdx);
if( ~strncmpi(opts.objFn,'trans',5) ) 
   warning('objective doesnt start with "trans".  Is it really a transfer-learning objective?  Rewritten...'); 
   opts.objFn=sprintf('trans%s',opts.objFn); 
end;
z=jf_cvtrain(z,opts.blockOptName,blockIdx,'objFn',opts.objFn,varargin{:});
return;
%--------------------------------------
function testCase()
nDs=5; 
clear zs; s2n=.01;
for dsi=1:nDs;
   zs(dsi)=jf_mksfToy(); 
   zs(dsi).X=repop(zs(dsi).X,'+',randn(size(zs(dsi).X,1),1)*10); % add data-set specific shift
                                                                 % data-set specific rotation
   [R(:,:,dsi),ans,ans] = svd(randn(size(zs(dsi).X,1))); % make random rotation matrix
   zs(dsi).X(:,:)   = (eye(size(zs(dsi).X,1))*s2n+(1-s2n)*R(:,:,dsi)) * zs(dsi).X(:,:);
end
z =jf_cat(zs,'dim','epoch');
%getBlockIdx(z)
jf_cvtrain(z)
jf_transcvtrain(z)

% diff rel strengths of learning
%    global ---------->  local   solutions
Cmmu=[1000 100   3    1    1   1;... % mag of the sub-prob reg
      1    1     1    1    3   100]; % mag of the mean reg
transCs=tprod(Cmmu(:,1:3),[1 3],10.^(3:-1:-3)',[2]); % [C x glob/loc x totalreg]  % only global solutions

jf_transcvtrain(z,'Cs',transCs)
