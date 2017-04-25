function [z]=jf_cv2train(z,varargin);
% double nested cross-validation based fitting of classifier model to data
z=jf_cvtrain(z,'cv2',1,varargin{:});
