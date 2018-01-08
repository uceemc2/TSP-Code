% Experimental results: function that computes the reconstruction PSNR. The
% variable "version" indicates the image that is considered. 
% The variable 'smax' indicate the maximum dimension
% of the affine subspaces corresponding to the GMM model adopted. 
addpath('./SyntheticData');
options = optimoptions('fmincon','MaxIter',80000,'MaxFunEvals',240000)
global g_Sigx1;
global g_Ri1;
global g_mux1;
global g_pcvett1;
global g_data1;
global g_w1;
global g_p1;
global g_p2;
global g_m;

global g_Phi1 ;
global g_Sigx1x2 ;
global g_Ri ;
global g_mux1x2 ;
global g_pcvett ;
global g_m1 ;
global g_m2 ;
global g_w;

version = 4;
smax = 68;
clc; 
close all; 

randn('state',0); rand('state',0); format short;
switch version
    case 1, dataset = 1; nameFig='pepper';
    case 2, dataset = 2; nameFig='barbara';
    case 3, dataset = 3; nameFig='house';
    case 4, dataset = 4; nameFig='lena';
    case 5, dataset = 5; nameFig='cameraman';
    otherwise, return;
end
% np=10; vettSNRdB=linspace(10,70,np); % vector containing the different noise levels considered
vettSNRdB = [40]; % vector containing the different noise levels considered, in this paper we assign noise level = -40 dB
np = 1; 

nrandom=1; % number of random realization of the measurment kernel over which the PSNR values are averaged

PatchSize_lar = 8; % the patch size of the large image
scale = 4;   % the scale to be resized for the small image. If the large image is 512x512, the small image will be 512/scale x 512/scale
images = {'pepper.png','barbara.png','house.png','lena.png','cameraman.png'};

vettm=10:10; % vector containing the number of measurements considered
m2=4;        % vector containing the number of measurements considered for side information

% train the GMM
% savename = Train_couple_GMM(Train_img_file, PatchSize_lar,scale);


savename='Model_SR_PL8_PS2.mat'; % load training data (obtained by EM algorithm)

% test the reconstruction
Test_img_file = images{dataset};

X0= imread(Test_img_file);
[Row, Col,rgb] = size(X0);
if(rgb >1)
X= im2double(rgb2gray(X0));
else
X= im2double(X0);
end
[Row, Col] = size(X);

%PatchSize_lar = 8; % Define the patchsize
PatchSize_sml = PatchSize_lar/scale; 

X_small = imresize(X,[Row/scale, Col/scale]);
data1 = im2col(X,[PatchSize_lar PatchSize_lar], 'distinct');  % Here you can use 'sliding'
data2 = im2col(X_small,[PatchSize_sml PatchSize_sml], 'distinct');  % Here you can use 'sliding'
[p1,n1] = size(data1); %  n number of samples
[p2,n2] = size(data2); %  n number of samples
data=[data1;data2];

%load the trained GMM (obtained via EM)
load(savename);
Sigx1x2=Sig;
mux1x2=Mu;
paix1x2=pai;
Sigx1=Sig_lar;
mux1=Mu_lar;
paix1=pai_lar;
Sigx2=Sig_sml;
mux2=Mu_sml;
paix2=pai_sml;

% GENERATES HERE THE PROJECTED IMAGES
[nrow ncol nc]=size(Sigx1x2);
Ri0=zeros(nrow);

[py_c classes]=MAPclassifier(data,Sigx1x2,mux1x2,eye(nrow),paix1x2,Ri0); 
% PROJECTION OF THE DATA ONTO LOW DIMENSIONAL UNION OF SUBSPACES

% retains only the first smax principal components of each class conditioned covariance matrix
% creates the corresponding projectors
[Sigx1x2]=reduceDim(Sigx1x2, smax);
for k=1:nc
    [U D] = eig(Sigx1x2(:,:,k));
    Ur=U(:,nrow-smax+1:end);
    Proj(:,:,k)=herm(Ur*Ur');
end

Sigx12 = zeros(p1,p2,nc);
Sigx21 = zeros(p2,p1,nc);
for k=1:nc
    Sigx1(:,:,k)=Sigx1x2(1:p1,1:p1,k);
    Sigx2(:,:,k)=Sigx1x2(p1+1:end,p1+1:end,k);
    Sigx12(:,:,k)=Sigx1x2(1:p1,p1+1:p1+p2,k); 
    Sigx21(:,:,k)=Sigx1x2(p1+1:end,1:p1,k);
end
mux1=mux1x2(1:p1,:);
mux2=mux1x2(p1+1:end,:);
paix1=paix1x2;
paix2=paix1x2;    

% data projection: projects the original image over the union of subspaces
% corresdponding to the low-rank GMM prior
for k=1:nc
    indclass=find(classes==k);
    datared(:,indclass)=mux1x2(:,k)*ones(1,length(indclass))+Proj(:,:,k)*(data(:,indclass)-mux1x2(:,k)*ones(1,length(indclass)));
end


data1red=datared(1:p1,:);
data2red=datared(p1+1:end,:);
% rebuilds the image
X_red=col2im(data1red,[PatchSize_lar PatchSize_lar],[Row Col], 'distinct');
X_small_red=col2im(data2red,[PatchSize_sml PatchSize_sml],[Row/scale Col/scale], 'distinct');
PSNRprojVsOriginal1=SS_PSNR_3D(X,X_red);
PSNRprojVsOriginal2=SS_PSNR_3D(X_small,X_small_red);
% substitutes the original image with the projected image
data1=data1red;
data2=data2red;
data=datared;
X=X_red;
X_small=X_small_red;
% save projected images
imwrite(X_red,[nameFig,num2str(smax),'.png'],'png')
imwrite(X_small_red,[nameFig,num2str(smax),'scale',num2str(scale),'.png'],'png')
%imshow(X_red)

Sigx2noisy=zeros(p2,p2,nc);
Sigx1x2noisy=zeros(p1+p2,p1+p2,nc);
for k=1:nc
    Sigx2noisy(:,:,k) = Sigx2(:,:,k)+(1e-8)*eye(p2);
end
for cc=1:nc
    Sigx1x2noisy(:,:,cc)=[Sigx1(:,:,cc) Sigx12(:,:,cc);Sigx21(:,:,cc) Sigx2noisy(:,:,cc)];
end

% Measurements Design based on gsvd analysis
for kkk=1:nc                        
            AA = (Sigx2noisy(:,:,kkk));
            BB = (Sigx2noisy(:,:,kkk)-Sigx21(:,:,kkk)*pinv(Sigx1(:,:,kkk))*Sigx12(:,:,kkk));                    
            A = herm(sqrtm(herm(AA)));
            B = herm(sqrtm(herm(BB)));                       

            [U,V,XX,C,S] = gsvd(A,B);
            rx1 = rank(Sigx1(:,:,kkk));
            rx2 = rank(Sigx2(:,:,kkk));
            rx  = rank(Sigx1x2(:,:,kkk));
            tmpR= rx1+rx2-rx;      
            tmpR = 4;
            tmpC= p2 - tmpR;
            tmpPhi2(:,:,kkk) = [zeros(tmpR,tmpC) eye(tmpR)]*inv(XX);
end

str1 = '';
str2 = '';   
str3 = '';  
for cc=1:nc
    str1 = sprintf('tmpPhi2(:,:,%d);',cc);
    str2 = strcat(str2,str1);
    str3 = sprintf('DesPhi_2(:,:) = [%s];',str2);   
end
eval(str3);
kk1     = nchoosek(80,4);
Probnum = combntns(1:80,4);

[nrow2 ncol2 nc2]=size(Sigx2noisy);
for k=1:nc
    my=eye(nrow2)*mux2(:,k);
    pdftemp = mvnpdf(data2.',my.',Sigx2noisy(:,:,k));
    like(k,:)=(paix2(k)*pdftemp.');
    loglike=log(like);
    loglike=like;
end
[li classesidx]=max(loglike);

Phi01=randn(vettm,p1);
Phi02=randn(m2,p2);

for kk=1:length(vettm)
    m = vettm(kk);        
    P=m; % trace constraint: tr(M*M') = number of rows!
    for k = 1:np
        snr=10.^(vettSNRdB(k)/10);    
        snr
        rho=1/sqrt(snr);
        Ri1=rho^2*eye(m);
        Ri2=rho^2*eye(m2);
        Ri=rho^2*eye(m+m2);
        w1 = rho*randn(m,n1);
        w2 = rho*randn(m2,n2);

        RiI = [Ri1 zeros(m,p2); zeros(p2,m) zeros(p2)];
        XRnosi = zeros(p1,n1,m,nrandom);
        XRsi = zeros(p1,n1,m,nrandom);
        XRboth = zeros(p1+p2,n1,m,nrandom);
        for nr=1:nrandom
            
            Phi1=Phi01(1:m,:);
            Phi1=sqrt(P)*Phi1./trace(herm(Phi1*Phi1'));
            y1 = Phi1*data1 + w1;

            Phi2=Phi02(1:m2,:);
            Phi2=sqrt(m2)*Phi2./sqrt(trace(Phi2*Phi2'));
            y2 = Phi2*data2 + w2;
            y=[y1; y2];
            Phi=[Phi1 zeros(size(Phi1,1),size(Phi2,2));...
            zeros(size(Phi2,1),size(Phi1,2))  Phi2];

            disp('Random Measurements - (1)')
            XRboth(:,:,k,nr) = GMM_CS_Inv_samePhi_Ri(y,Phi,Sigx1x2,mux1x2,paix1x2,Ri);
            XRsi(:,:,k,nr)=XRboth(1:p1,:,k,nr);
            ImRsi = col2im(XRsi(:,:,k,nr),[PatchSize_lar PatchSize_lar],[Row Col], 'distinct');
            PSNRsi(nr,k) = SS_PSNR_3D(X,ImRsi);
            PSNRsi(nr,k);
            imwrite(ImRsi,[nameFig,'_smax=',num2str(smax),'_M1=',num2str(m),'_SNR=',num2str(vettSNRdB(k)),'_WithSI_Random.png'],'png')
                        
            % ***********************************************************
            % Optimize Phi2 numerically
            % ***********************************************************
            disp('Measurements design: numerically - (2)')
            g_Phi1 = Phi1;
            g_Sigx1x2 = Sigx1x2;
            g_Ri = Ri;
            g_mux1x2 = mux1x2;
            g_pcvett = paix1x2;
            g_data1 = [data1;data2];
            g_w = [w1;w2];
            g_p1 = p1;
            g_p2 = p2;
            g_m1 = m;
            g_m2 = m2;
            optPhi2 = sym('optPhi2',[size(Phi2)]);
            x0=Phi2;
            A=[];b=[];Aeq=[];beq=[];lb=[];ub=[];
            [c,ceq] = mycon2(optPhi2);            
            [optPhi2,fval,exitflag,output] = fmincon(@myfunonlyphi2_img_si,x0,A,b,Aeq,beq,lb,ub,@mycon,options);            
            
            Phi_new = [Phi1 zeros(m,p2); zeros(m2,p1) optPhi2];
            y_si=Phi_new*[data1;data2]+[w1;w2];
            XRboth(:,:,k,nr) = GMM_CS_Inv_samePhi_Ri(y_si,Phi_new,Sigx1x2,mux1x2,paix1x2,Ri);
            XRsi(:,:,k,nr)=XRboth(1:p1,:,k,nr);
            ImRsi5 = col2im(XRsi(:,:,k,nr),[PatchSize_lar PatchSize_lar],[Row Col], 'distinct');
            PSNRsi5(nr,k) = SS_PSNR_3D(X,ImRsi5); 
            imwrite(ImRsi5,[nameFig,'_smax=',num2str(smax),'_M1=',num2str(m),'_SNR=',num2str(vettSNRdB(k)),'_WithSI_Numberical.png'],'png')
                   
          % *************************************************************
          %   measurements design: based on gsvd analysis
          % *************************************************************
          disp('Measurements (sub)design: Based on gsvd analysis - (3)')             
          for i = 1:kk1
%           for i = 1:100
                r2 = rho*randn(m2,n2);
                tmpclass = Probnum(i,:);
                tmp_1    = DesPhi_2(tmpclass(1,1),:);
                tmp_2    = DesPhi_2(tmpclass(1,2),:);
                tmp_3    = DesPhi_2(tmpclass(1,3),:);
                tmp_4    = DesPhi_2(tmpclass(1,4),:);
                testPhi2 = [tmp_1;tmp_2;tmp_3;tmp_4];
                Phi=[Phi1 zeros(size(Phi1,1),size(Phi2,2));...
                    zeros(size(Phi2,1),size(Phi1,2))  testPhi2];
                
                y2 = testPhi2*data2 + r2;
                y=[y1; y2];                
                XRboth(:,:,k,nr) = GMM_CS_Inv_samePhi_Ri(y,Phi,Sigx1x2,mux1x2,paix1x2,Ri);
                XRsi(:,:,k,nr)=XRboth(1:p1,:,k,nr);
                ImRsi = col2im(XRsi(:,:,k,nr),[PatchSize_lar PatchSize_lar],[Row Col], 'distinct');
                PSNRsi6(i,:) = SS_PSNR_3D(X,ImRsi);
            end
            row_index = find(PSNRsi6 == max(PSNRsi6));
            PSNRsi_6(nr,k) = PSNRsi6(row_index,:);
            
            tmpclass = Probnum(row_index,:);
            tmp_1    = DesPhi_2(tmpclass(1,1),:);
            tmp_2    = DesPhi_2(tmpclass(1,2),:);
            tmp_3    = DesPhi_2(tmpclass(1,3),:);
            tmp_4    = DesPhi_2(tmpclass(1,4),:);
            testPhi2 = [tmp_1;tmp_2;tmp_3;tmp_4];
            Phi=[Phi1 zeros(size(Phi1,1),size(Phi2,2));...
                 zeros(size(Phi2,1),size(Phi1,2))  testPhi2]; 
            y2 = testPhi2*data2 + r2;
            y=[y1; y2]; 
            XRboth(:,:,k,nr) = GMM_CS_Inv_samePhi_Ri(y,Phi,Sigx1x2,mux1x2,paix1x2,Ri);
            XRsi6(:,:,k,nr)=XRboth(1:p1,:,k,nr);
            ImRsi6 = col2im(XRsi6(:,:,k,nr),[PatchSize_lar PatchSize_lar],[Row Col], 'distinct');                  
            imwrite(ImRsi6,[nameFig,'_smax=',num2str(smax),'_M1=',num2str(m),'_SNR=',num2str(vettSNRdB(k)),'_WithSI_Gsvd.png'],'png')
        end
      % *****************************************************************
      % PSNR results for: 1) random measurements; 2) optimal measusmrents
      % (numerially); and 3) optimal meaurements (gsvd design)
      % *****************************************************************
        PSNRRandom(kk,k)           =mean(PSNRsi(:,k))   
        PSNROptimalNumerical(kk,k) =mean(PSNRsi5(:,k))
        PSNROptimalGsvd(kk,k)      =mean(PSNRsi_6(:,k))
        %disp(['n. of measurements = ' num2str(m) ', \sigma^2 = ' num2str(10*log10(rho.^2)) ' dB, dataset ' num2str(dataset) ', random averaged over ' num2str(nrandom) ' realizations.']);        
    end
end
