function [f] = myfunmulti_phi2(inputPh2)%[f, gradf] = myfunmulti(inputPhi1)
global g_M1;
global g_M2;
global g_Phi1;
global g_Sigx1x2;
global g_Ri;
global g_mux1x2;
global g_pcvett;
global g_data1;
global g_w;
global g_nx1;
global g_nx2;

Phinew = [g_Phi1 zeros(g_M1,g_nx2); zeros(g_M2,g_nx1) inputPh2];
y = Phinew*g_data1 + g_w;

[nvett nobs]=size(y);
[ny nx]=size(Phinew);

pc   = g_pcvett;
MuC  = g_mux1x2;
Ri   = g_Ri;
SigC = g_Sigx1x2;
M    = Phinew;
for k=1:length(pc)
    myC = M*MuC(:,k);
    pdftemp = mvnpdf(y.',myC.',herm(Ri+M*SigC(:,:,k)*M'));
    pdfGweightedMult(:,:,k) = pc(k)*repmat(pdftemp.',nx,1);
    vetti(:,:,k) = SigC(:,:,k)*M'*(herm(Ri+M*SigC(:,:,k)*M')\(y-repmat(myC,1,nobs)))+repmat(MuC(:,k),1,nobs);
    termi(:,:,k) = pdfGweightedMult(:,:,k).*vetti(:,:,k);    
end
sumpdfGwiegthed=sum(pdfGweightedMult,3);
R_X = sum(termi,3)./sumpdfGwiegthed;

opt_xhat_si=R_X(1:g_nx1,:);
f = mean(sum(abs(opt_xhat_si-g_data1(1:g_nx1,:)).^2));