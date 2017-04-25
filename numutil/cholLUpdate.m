function [r,err]=cholLUpdate(chol,L)
% Compute the new row required to update the cholesky factorisation when 
% we add a lower right L shape to the target matrix, 
% i.e. A' = [ A L(1:end-1)'; L];
   r=zeros(1,numel(L)); r2=0;
   if ( ~isempty(chol) ) %test the edge case for the first call with empty chol
      r(1:end-1)=(chol\L(1:end-1))'; r2=r(1:end-1)*r(1:end-1)';
   end
   r(end)=real(sqrt( L(end) - r2 ));     

   % test for rounding errors which indicate the factorisation has broken down 
   % and we should re-build it!
   if ( L(end) - r2 < 0 ) 
      err='round'; return; 
   end
   
   % test for linear dependence, by estimating the condition number.  
   % N.B. as chol is triangular the rank estimate is simply the number of 
   % non-zero columns of chol_covXaXa OR the ratio of new diag 
   % element, t, to largest previous one.  Note, both these tests are 
   % *only* valid if chol is square!
   % further that the actual condition number of covXaXa is this squared!
   % new dim makes covXaXa singular, so permentaly remove it.
   % As chol effectively diagonalises A we can think of the rows of chol as 
   % the eigenvectors of A.  Hence A is singular if any 2 of these 
   % eigenvectors are parallel, so we test for singularity by computing and 
   % thresholding the inner-products.   
   if( abs(r(end))./max(abs(diag(chol))) < 1e-2 )
      if ( any(1-r(1:end-1)*chol'./(sqrt(sum(chol.*chol,2))'*norm(r(1:end-1))) < 1e-8) )
         %      sum([sum(abs(chol),1) r(end)]>1e-5) < size(chol,1)+1 ) 
         err='singular'; return;      
      end  
   end
   
   % Everything's OK so return the new row
   %newR=[chol zeros(size(chol,1),1);r];
   err=[];
