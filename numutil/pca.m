clear;

% example for principal components in matlab
sDir='../data/xrce/books/';
sPrefix='books';
sExtn='jpg';

nimage=10;
sthing='20';


A=double(imread([sDir,sPrefix,sthing,'.',sExtn],sExtn));
if ( ndims(A) > 2 ) A=mean(A,3); end %convert to grayscale
[m,n]=size(A)

nn=m*n;

X=zeros(nimage,nn);

size(X)
size(A)

X(1,:)=A(:)';

% read the images from one group of things
for i=2:nimage
    A=double(imread([sDir,sPrefix,num2str(i+9,'%02d'),'.',sExtn],sExtn));
    if ( ndims(A) > 2 ) A=mean(A,3); end    % convert to gray-scale
    
    % make vector from matrix
    X(i,:)=A(:)';
end


% compute the correlation matrix
Z=zeros(size(X));
for i=1:nimage
% compute the standardized, centralized vectors in Z
    Z(i,:)=(X(i,:)-mean(X(i,:)))/std(X(i,:));
end
K=Z*Z';  % correlation between the images

% compute the eigen vectors U and the eigenvalues V
[U,V]=eig(K);


% compute the principal components
Y=U'*Z;

W=zeros(size(X));

% compute a 0-255 scaled gray image
for i=1:nimage
    W(i,:)=(Y(i,:)-min(Y(i,:)))*(255/(max(Y(i,:))-min(Y(i,:))));
end

% display the principal components
for i=1:nimage
    B=reshape(W(i,:),m,n);
    image(B);
    colormap(gray(256));
    pause;
end

