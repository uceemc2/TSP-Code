% MAP classifier that deteremines the class of the GMM input

function [maxpc_y cest]=MAPclassifier(y,KxC,mxc,M,pc,Ri)

[ny nobs]=size(y);

for k=1:length(pc)
    my=M*mxc(:,k);
    pdftemp=mvnpdf(y.',my.',herm(Ri+M*KxC(:,:,k)*M'));
    like(k,:)=(pc(k)*pdftemp.');
    loglike=log(like);
    loglike=like;
end

[li cest]=max(loglike);

% posterior probabilities

py=sum(like);
pc_y=like./repmat(py,length(pc),1);
[maxpc_y index]=max(pc_y);
