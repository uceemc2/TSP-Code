function f = myfunonlyphi2_img_si(inputPhi2)
global g_Phi1;
global g_Sigx1x2;
global g_Ri;
global g_mux1x2;
global g_pcvett;
global g_data1;
global g_w;
global g_p1;
global g_p2;
global g_m1;
global g_m2;

Phinew = [g_Phi1 zeros(g_m1,g_p2);zeros(g_m2,g_p1) inputPhi2];
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

opt_xhat_si=R_X;
f = mean(sum(abs(opt_xhat_si-g_data1).^2));