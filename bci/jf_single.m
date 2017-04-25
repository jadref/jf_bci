function z=jf_single(z);
% convert double data to single
if( isa(z.X,'double') ) z.X=single(z.X); end;
if( isfield(z,'Y') && isa(z.Y,'double') ) z.Y=single(z.Y); end;
z=jf_addprep(z,mfilename,'',[],[]);
return;