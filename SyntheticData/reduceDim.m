% function to reduce the dimensionality (smax) of the GMM model

function [Sigout]=reduceDim(Sigin, smax)

[nr nc nK]=size(Sigin);

Sigout=zeros(nr,nc,nK);

for k=1:nK
    
    [U D]=eig(Sigin(:,:,k));
    D(1:nc-smax,1:nc-smax)=0;
    Sigout(:,:,k)=herm(U*D*U');
end
