function [dvseq]=dv2seqdv(dv,dim,seqs)
% convert per-epoch decision values into sequence decision values
%
% Inputs:
%  dv -- [n-d float] set of decision values
%  dim-- [int] sequence dimenaion of dv
%  seq-- [size(dv) x M] set of possible sequences. 
%        N.B. any dim can be singlenton which implies the value is constant over this dim
% Outputs:
%  dvseq -- [size(dv) except size(dv,dim)==M] set of sequence decision values
dvDimSpec=[1:dim-1 -dim dim+1:ndims(dv)];
seqDimSpec=[1:dim-1 -dim dim+1:ndims(seqs)-1 dim]; szSeq=size(seqs); seqDimSpec(szSeq==1)=0;
dvseq = tprod(dv,dvDimSpec,seqs,seqDimSpec);
return;
%----------------------------------------------------------------------
function testCase()
nEpoch=10; nLetter=10; nSeq=5;
dv =sign(randn(nEpoch,nLetter)); % epoch x letter
seq=sign(randn(nEpoch,1,nSeq));% epoch x letter x position
[dvseq]=dv2seqdv(dv,1,seq);    % position x letter