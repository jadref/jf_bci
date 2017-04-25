function [chol,err]=cholXdel(chol,j);
% update a cholesky factorisation to delete the + shaped row and column j
% N.B. this _*doesn't*_ resize the array, just move entries in it, 
% so the last row/col will contain bollocks!

% To downdate cholesky factorisation we first remove the row 
chol(j:end-1,1:j-1)=chol(j+1:end,1:j-1);
% and then use a rank-1 *update* to include the effects of the now surplus j'th
% column in the rest of the factorisation.
% Note: because col j is 0 above row j, only need to update the lower 
% right block of chol_covXaXa
[chol(j:end-1,j:end-1),err]=...
    cholRank1Update(chol(j+1:end,j+1:end),chol(j+1:end,j));
