% GMM conditional mean estimator (you can do it all in one,
% putting as input the matrix containing all the observation vectors,
% using the function mvnpdf )

function R_X = GMMcondMeanEstColorNoise(y,SigC,MuC,pc,M,Ri)

[nvett nobs]=size(y);
[ny nx]=size(M);
iRi=inv(Ri);

for k=1:length(pc) %pc-2classes
    % VALID ALSO FOR RANK DEFICIENT COVARIANCE MATRICES
    myC = M*MuC(:,k);
    pdftemp = mvnpdf(y.',myC.',herm(Ri+M*SigC(:,:,k)*M'));
    pdfGweightedMult(:,:,k) = pc(k)*repmat(pdftemp.',nx,1);
    vetti(:,:,k) = SigC(:,:,k)*M'*(herm(Ri+M*SigC(:,:,k)*M')\(y-repmat(myC,1,nobs)))+repmat(MuC(:,k),1,nobs);
    termi(:,:,k) = pdfGweightedMult(:,:,k).*vetti(:,:,k);
end
sumpdfGwiegthed=sum(pdfGweightedMult,3);
R_X = sum(termi,3)./sumpdfGwiegthed;


% TEST = pdfGweightedMult/sumpdfGwiegthed