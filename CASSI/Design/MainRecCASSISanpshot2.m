savename = 'Model_TRAIN6IMAGES_RGB_512_HSI_256_K20_TEST_FRO_S1_Ch33.mat';
K1K2=20;
NormPhi = 1;
filename = 'scene1';
nrandom = 1; 

% ************************
olp_lar = 8;
olp = 4; 
% ************************

% load('tmpPhiHsiCASSI.mat');

mea_HSI = 1;
mea_rgb = 'rgb';
resize_flag = 1;
Row_hsi = 256;
Col_hsi = 256;
PatchSize = 4;

Row_rgb = 512
Col_rgb = 512;
PatchSize_lar = 8;

% Read image, you can use many different images to train the dictionary.
addpath(filename);

% load the HSI data
list_hsi = dir([filename '/*_reg1.mat']);
load(list_hsi.name);
x_hsi0 = reflectances;
clear reflectances list_hsi
% the name is reflectances

% read the rgb file
list_rgb = dir([filename '/*.bmp']);
x_rgb0 = im2double(imread(list_rgb.name));
clear list_rgb


%figure; imagesc(X); colormap gray
[Row, Col, Ch] = size(x_hsi0);
[Row, Col, rgb] = size(x_rgb0);


if(resize_flag)
for nc = 1:Ch
    x_hsi(:,:,nc) = imresize(x_hsi0(:,:,nc) , [Row_hsi,Col_hsi]);
end

for nc = 1:rgb
    x_rgb(:,:,nc) = imresize(x_rgb0(:,:,nc) , [Row_rgb,Col_rgb]);
end
end

[Row, Col, Ch] = size(x_hsi);

%load the trained GMM
load(savename);
Row_new = floor(Row/PatchSize)*PatchSize;
Col_new = floor(Col/PatchSize)*PatchSize;
x_hsi = x_hsi(1:Row_new, 1:Col_new,:);

[Row, Col, rgb] = size(x_rgb);
Row_new_rgb = floor(Row/PatchSize_lar)*PatchSize_lar;
Col_new_rgb = floor(Col/PatchSize_lar)*PatchSize_lar;
x_rgb = x_rgb(1:Row_new_rgb, 1:Col_new_rgb,:);

for mea_HSI=1:1 
    kk = mea_HSI;
    
    % generate the Phi_matrix with the CASSI scenario
    T = mea_HSI;
    for t= 1:T
        Phi{t} = binornd(1,0.5,[Row_new, Col_new,Ch]);
    end
    shift = 1;
    shift_T = 17;
    for t1=1:T
        if(t1>1)
            Phi{t1}(:,1+(t1-1)*shift_T:Col_new,1) = Phi{t1-1}(:,1+(t1-2)*shift_T:(Col_new-shift_T),1);
        end
        for t=2:Ch
            Phi{t1}(1+(t-1)*shift:Row_new,:,t) = Phi{t1}(1+(t-2)*shift:Row_new-shift,:,t-1); 
        end
    end

% now, in order for the fast inversion, for each patch, we use the same
% mask
begin_P =Ch*2+shift+shift_T;
Kron_unit_row = Row_new/PatchSize;
Kron_unit_col = Col_new/PatchSize;
for t1= 1:T
    for t=1:Ch
        Phi_new_unit{t1}(:,:,t) =  Phi{t1}(begin_P+(1:PatchSize),begin_P+(1:PatchSize),t);
        Phi_new{t1}(:,:,t) = kron(ones(Kron_unit_row, Kron_unit_col),Phi_new_unit{t1}(:,:,t));
    end
end
clear Phi 

for t1 = 1:T 
    y_hsi{t1} = sum(x_hsi.*Phi_new{t1},3);
end

% now we get the measurement of the rgb image
if(strcmp(mea_rgb,'rgb'))
    y_rgb = x_rgb;
    Phi_rgb_p = sparse(eye(PatchSize^2*rgb));
end

% olp = PatchSize; % here we use non-overlapping patches
% olp_lar=PatchSize_lar;

% vectorize the patches
y_hsi_all = reshape(cell2mat(y_hsi),[Row_new, Col_new,T]);
y_hsi_p = video2patches_fast(y_hsi_all,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'
x_hsi_p = video2patches_fast(x_hsi,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'
x_rgb_p = video2patches_fast(x_rgb,PatchSize_lar, PatchSize_lar,olp_lar,olp_lar);  % Here you can use 'sliding'

Cr=0.2126;
Cg=0.7152;
Cb=0.0722;

Phi_Cr = Cr*eye(PatchSize_lar*PatchSize_lar);
Phi_Cg = Cg*eye(PatchSize_lar*PatchSize_lar);
Phi_Cb = Cb*eye(PatchSize_lar*PatchSize_lar);

Phi_rgb_p = [Phi_Cr Phi_Cg Phi_Cb]; 
Phi_hsi   = getw_unit(Phi_new_unit);

% load('Overlapping_PhiHsi.mat');
%****************************************************
% Fixed measurements kernel by function CASSIdesign()
%****************************************************
    % load('RandomPhi2_16.mat');   % 25.7831 dB
    % load('RandomPhi2_32.mat');   % 28.6267 dB
    % load('FinalPhi2_16.mat');    % 28.3930 dB
    % load('FinalPhi2_32_02.mat'); % 30.7755 dB
    for nr=1:nrandom
        Phi_both = [Phi_rgb_p zeros(size(Phi_rgb_p,1),size(Phi_hsi,2));...
            zeros(size(Phi_hsi,1), size(Phi_rgb_p,2))  Phi_hsi];
        
        [M2 n2]=size(Phi_hsi)
        r2=rank(Phi_hsi)
        [M1 n1]=size(Phi_rgb_p)
        
%         figure(10)         
%         test_phi = Phi_hsi(1:16,1:128);
%         imagesc(test_phi)
%         imagesc(Phi_hsi)
%         figure(11)
%         imagesc(Phi_rgb_p)

        X_rec_hsi = zeros(size(x_hsi_p));
        X_rec_rgb = zeros(size(x_rgb_p));
        y_rgb_p = Phi_rgb_p*x_rgb_p;

        y_hsi_p = Phi_hsi*x_hsi_p;
        X_rec_both = GMM_CS_Inv_samePhi([y_rgb_p;y_hsi_p],Phi_both,Sig,Mu,pai);        

        image_recon_joint_RGB = patches2video_fast(X_rec_both(1:PatchSize_lar^2*rgb,:), Row_rgb, Col_rgb,rgb, olp_lar, olp_lar);
        RECNAME ='';
        RECNAME = sprintf('CASSI_design.png',kk);
        imwrite(image_recon_joint_RGB,RECNAME,'png');

        PSNR_joint_RGB(nr,kk)  = SS_PSNR_3D(x_rgb_p,X_rec_both(1:PatchSize_lar^2*rgb,:));
        PSNR_joint_RGB
    end
end
imwrite(x_rgb,'RGBoriginal.png','png');


% new_y_hsi_p = new_Phi_hsi(:,:,row_index)*x_hsi_p;
% new_Phi_both = [Phi_rgb_p zeros(size(Phi_rgb_p,1),size(new_Phi_hsi(:,:,row_index),2));...
%                 zeros(size(new_Phi_hsi(:,:,row_index),1), size(Phi_rgb_p,2))  new_Phi_hsi(:,:,row_index)];
% new_X_rec_both = GMM_CS_Inv_samePhi([y_rgb_p;new_y_hsi_p],new_Phi_both,Sig,Mu,pai);
% image_recon_joint_RGB_opt = patches2video_fast(new_X_rec_both(1:PatchSize_lar^2*rgb,:), Row_rgb, Col_rgb,rgb, olp_lar, olp_lar);
% 
% image_recon_joint_RGB_opt = patches2video_fast(new_X_rec_both(1:PatchSize_lar^2*rgb,:), Row_rgb, Col_rgb,rgb, olp_lar, olp_lar);

%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
% % Design case of dsvd-based Phi2
% for cc=1:K1K2
%     Sigx12(:,:,cc) = Sig(1:192,192+1:192+528,cc);
%     Sigx21(:,:,cc) = Sig(192+1:end,1:192,cc);
%     Sigx1(:,:,cc)  = Sig(1:192,1:192,cc);
%     Sigx2(:,:,cc)  = Sig(192+1:192+528,192+1:192+528,cc);        
% %     Sigx12(:,:,cc) = Sig(1:48,48+1:48+528,cc);
% %     Sigx21(:,:,cc) = Sig(48+1:end,1:48,cc);
% %     Sigx1(:,:,cc)  = Sig(1:48,1:48,cc);
% %     Sigx2(:,:,cc)  = Sig(48+1:48+528,48+1:48+528,cc);    
% end
% % % % % xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
% testM2 = 16;
% tmpPhi2 = zeros(testM2,528,K1K2);
% for cc=1:K1K2
%             AA = (Sigx2(:,:,cc));
%             BB = (Sigx2(:,:,cc)-Sigx21(:,:,cc)*pinv(Sigx1(:,:,cc))*Sigx12(:,:,cc));
%             A = [];
%             B = [];
%             A = herm(sqrtm(herm(AA)));
%             B = herm(sqrtm(herm(BB)));
%             U = []; V=[]; X=[]; C=[]; S =[];
%             [U,V,X,C,S] = gsvd(A,B,0);
%             rx1 = rank(Sigx1(:,:,cc));
%             rx2 = rank(Sigx2(:,:,cc));
%             rx  = rank(Sig(:,:,cc));
%             tmpR= max(testM2,rx1+rx2-rx);
%             tmpC= 528 - tmpR;
%             tmpPhi2(:,:,cc) = [zeros(tmpR,tmpC) eye(tmpR)]*inv(X);
% end
% 
% str1 = '';
% str2 = '';    
% str3 = '';
% DesPhi_2 = zeros(K1K2*testM2,528);
% for cc=1:K1K2
%     str1 = sprintf('tmpPhi2(:,:,%d);',cc);
%     str2 = strcat(str2,str1);
%     str3 = sprintf('DesPhi_2(:,:) = [%s];',str2);   
% end
% eval(str3);
% 
% phi1tmpI = eye(K1K2*testM2);
% 
% TestNum = 3000;
% I_k = zeros(TestNum,testM2);
% for j=1:TestNum    
%     I_1 = randperm(K1K2*testM2);
%     I_k(j,:) = I_1(1:testM2);
%     
%     Phi1tmp = phi1tmpI(I_k(j,:),:);
%     new_Phi_hsi = Phi1tmp*DesPhi_2;
%     
%     new_y_hsi_p = new_Phi_hsi*x_hsi_p;
%     new_Phi_both = [Phi_rgb_p zeros(size(Phi_rgb_p,1),size(new_Phi_hsi,2));...
%                 zeros(size(new_Phi_hsi,1), size(Phi_rgb_p,2))  new_Phi_hsi];
%     new_X_rec_both = GMM_CS_Inv_samePhi([y_rgb_p;new_y_hsi_p],new_Phi_both,Sig,Mu,pai);  
%     new_PSNR_joint_RGB(j,:) = SS_PSNR_3D(x_rgb_p,new_X_rec_both(1:PatchSize_lar^2*rgb,:));
%     j
%     new_PSNR_joint_RGB(j,:)
% end
% row_index = find(new_PSNR_joint_RGB == max(new_PSNR_joint_RGB));
% Phi1tmp = phi1tmpI(I_k(row_index,:),:);
% new_Phi_hsi = Phi1tmp*DesPhi_2;
% new_y_hsi_p = new_Phi_hsi*x_hsi_p;
% new_Phi_both = [Phi_rgb_p zeros(size(Phi_rgb_p,1),size(new_Phi_hsi,2));...
%                 zeros(size(new_Phi_hsi,1), size(Phi_rgb_p,2))  new_Phi_hsi];
% new_X_rec_both = GMM_CS_Inv_samePhi([y_rgb_p;new_y_hsi_p],new_Phi_both,Sig,Mu,pai);
% 
% image_recon_joint_RGB_opt = patches2video_fast(new_X_rec_both(1:PatchSize_lar^2*rgb,:), Row_rgb, Col_rgb,rgb, olp_lar, olp_lar);
% RECNAME ='';
% RECNAME = sprintf('GSVDopt.png',kk);
% imwrite(image_recon_joint_RGB_opt,RECNAME,'png');
% new_PSNR_joint_RGB(row_index,:)
% PSNR_joint_RGB_20_opt(kk,:) = new_PSNR_joint_RGB(row_index,:)
% 
% PhiGSVD = new_Phi_hsi;
% clear new_Phi_hsi;

%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX END GSVD

% %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Optimal CASSI(III)
% Load GSVD Phi2
% load('tmpPhiGSVD_M16K20_tset1.mat');
% temp_Phi_hsi = zeros(36,4);
% load('tmpPhiGSVD_1_M32K20.mat');
% PhiGSVD = Phi_hsi;
% % Closest Design for CASSI
% for j=1:500
%     
%  if kk == 1
%     temp_Phi_hsi = binornd(1,0.5,[36,4]);   
%     temp1 = temp_Phi_hsi(33:36,:);
%     temp = diag(temp1(:));
%     for ch=2:33
%         temp1 =  temp_Phi_hsi(33-ch+1:36-ch+1,:);
%         temp = [temp diag(temp1(:))];
%     end
%     new_Phi_hsi(:,:,j) = temp;
%     clear temp
%     clear temp1
%         
%     mindis(j,:) = norm(new_Phi_hsi(:,:,j)'-PhiGSVD','fro');
%     mindis(j,:)  
%   
%     new_y_hsi_p = new_Phi_hsi(:,:,j)*x_hsi_p;
%     new_Phi_both = [Phi_rgb_p zeros(size(Phi_rgb_p,1),size(new_Phi_hsi(:,:,j),2));...
%                 zeros(size(new_Phi_hsi(:,:,j),1), size(Phi_rgb_p,2))  new_Phi_hsi(:,:,j)];
%     new_X_rec_both = GMM_CS_Inv_samePhi([y_rgb_p;new_y_hsi_p],new_Phi_both,Sig,Mu,pai);  
%     new_PSNR_joint_RGB(j,:) = SS_PSNR_3D(x_rgb_p,new_X_rec_both(1:PatchSize_lar^2*rgb,:));
%     new_PSNR_joint_RGB(j,:)   
%  end
%     
%  if kk == 2
%     Ch = 33;
%     for t= 1:T
%         Phi{t} = binornd(1,0.5,[Row_new, Col_new,Ch]);
%     end
%     shift = 1;
%     shift_T = 17;
%     for t1=1:T
%         if(t1>1)
%         Phi{t1}(:,1+(t1-1)*shift_T:Col_new,1) = Phi{t1-1}(:,1+(t1-2)*shift_T:(Col_new-shift_T),1);
%         end
%         for t=2:33
%             Phi{t1}(1+(t-1)*shift:Row_new,:,t) = Phi{t1}(1+(t-2)*shift:Row_new-shift,:,t-1); 
%         end
%     end     
%     begin_P =Ch*2+shift+shift_T;
%     Kron_unit_row = Row_new/PatchSize;
%     Kron_unit_col = Col_new/PatchSize;
%     for t1= 1:T
%         for t=1:Ch
%         Phi_new_unit{t1}(:,:,t) =  Phi{t1}(begin_P+(1:PatchSize),begin_P+(1:PatchSize),t);
%         Phi_new{t1}(:,:,t) = kron(ones(Kron_unit_row, Kron_unit_col),Phi_new_unit{t1}(:,:,t));
%         end
%     end
%     clear Phi     
%     new_Phi_hsi(:,:,j)= getw_unit(Phi_new_unit);
%     
%     new_y_hsi_p = new_Phi_hsi(:,:,j)*x_hsi_p;
%     new_Phi_both = [Phi_rgb_p zeros(size(Phi_rgb_p,1),size(new_Phi_hsi(:,:,j),2));...
%                 zeros(size(new_Phi_hsi(:,:,j),1), size(Phi_rgb_p,2))  new_Phi_hsi(:,:,j)];
%     new_X_rec_both = GMM_CS_Inv_samePhi([y_rgb_p;new_y_hsi_p],new_Phi_both,Sig,Mu,pai);  
%     new_PSNR_joint_RGB(j,:) = SS_PSNR_3D(x_rgb_p,new_X_rec_both(1:PatchSize_lar^2*rgb,:));
%     j
%     new_PSNR_joint_RGB(j,:)  
% 
%  end
% end
% row_index = find(new_PSNR_joint_RGB == max(new_PSNR_joint_RGB));
% new_PSNR_joint_RGB(row_index,:)
% 
% new_y_hsi_p = new_Phi_hsi(:,:,row_index)*x_hsi_p;
% new_Phi_both = [Phi_rgb_p zeros(size(Phi_rgb_p,1),size(new_Phi_hsi(:,:,row_index),2));...
%                 zeros(size(new_Phi_hsi(:,:,row_index),1), size(Phi_rgb_p,2))  new_Phi_hsi(:,:,row_index)];
% new_X_rec_both = GMM_CS_Inv_samePhi([y_rgb_p;new_y_hsi_p],new_Phi_both,Sig,Mu,pai);
% image_recon_joint_RGB_opt = patches2video_fast(new_X_rec_both(1:PatchSize_lar^2*rgb,:), Row_rgb, Col_rgb,rgb, olp_lar, olp_lar);
% 
% RECNAME ='';
% RECNAME = sprintf('CASSIopt.png',kk);
% imwrite(image_recon_joint_RGB_opt,RECNAME,'png');

%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX END Optimal CASSI
% clear new_Phi_hsi;
% clear mindis;

% % SLM with interval [0,1]
% Phi_hsi_CASSI_opt = Phi_hsi_CASSI * GrayTranM; % interval [0,1]
% opt_y_hsi_p = Phi_hsi_CASSI_opt*x_hsi_p;
% opt_Phi_both = [Phi_rgb_p zeros(size(Phi_rgb_p,1),size(Phi_hsi_CASSI_opt,2));...
%                 zeros(size(Phi_hsi_CASSI_opt,1), size(Phi_rgb_p,2))  Phi_hsi_CASSI_opt];
% opt_X_rec_both = GMM_CS_Inv_samePhi([y_rgb_p;opt_y_hsi_p],opt_Phi_both,Sig,Mu,pai);
% 
% opt_PSNR_joint_RGB(nr,kk)  = SS_PSNR_3D(x_rgb_p,opt_X_rec_both(1:PatchSize_lar^2*rgb,:));
% 
% if K1K2==10
% PSNR_joint_RGB_10_test(kk,:) = opt_PSNR_joint_RGB(nr,kk);
% PSNR_joint_RGB_10_test
% end
% 
% if K1K2==20
% PSNR_joint_RGB_20_test(kk,:) = opt_PSNR_joint_RGB(nr,kk);
% PSNR_joint_RGB_20_test
% end
% 
% if K1K2==30
% PSNR_joint_RGB_30_test(kk,:) = opt_PSNR_joint_RGB(nr,kk);
% PSNR_joint_RGB_30_test
% end

% CASSISLMmeasuremets = patches2video_fast(opt_y_hsi_p, 256, 256,1, 4, 4);
% imwrite(CASSISLMmeasuremets,'EXPD/CASSISLMmeasuremets.png','png');

%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX END Optimal CASSI (II)

% end

% X_rec_woSI = GMM_CS_Inv_samePhi(y_rgb_p,Phi_rgb_p,Sig_rgb,Mu_rgb,pai_rgb);
% PSNR_woSI_RGB = SS_PSNR_3D(x_rgb_p,X_rec_woSI);
% PSNR_woSI_RGB
% image_recon_woSI_RGB = patches2video_fast(X_rec_woSI, Row_rgb, Col_rgb,rgb, olp_lar, olp_lar);
% imwrite(x_rgb,'RGBwoSI.png','png');


% end


% Wvls = 400:10:720;
% vects=6; vects2=6; mid = 3;
% dispCubeAshwin(x_hsi,1,Wvls,[],vects,vects2,0,1,'truth')
% figure; 
% for m = 1:mea_HSI
% subplot(1,mea_HSI,m); imshow(y_hsi{m}./max(max(y_hsi{m})))
% end
% 
% 
% opt_y_hsi = patches2video_fast(opt_y_hsi_p, 256, 256,1, 4, 4);
% 
% figure; 
% for m = 1:mea_HSI
% subplot(1,mea_HSI,1); imshow(opt_y_hsi./max(max(opt_y_hsi)))
% end

% dispCubeAshwin(image_recon_hsi,1,Wvls,[],vects,vects2,0,1,'wo_side')
% dispCubeAshwin(image_recon_joint_hsi,1,Wvls,[],vects,vects2,0,1,'w_side')
% figure; subplot(121); imshow(image_recon_lar); title(['W/O side, PSNR: ' num2str(PSNR)]);
% subplot(122);mea_HSI imshow(image_recon_joint_lar); title(['W side, PSNR: ' num2str(PSNR_joint)]);

% Rec.img_wo_side = image_recon_hsi; % the reconstructed image w/o side information
% Rec.X_hsi = x_hsi; % the truth large iamge
% Rec.X_rgb = x_rgb; % the truth large iamge
% Rec.img_w_side = image_recon_joint_hsi;
% Rec.PSNR_wo_side = PSNR;
% Rec.PSNR_w_side = PSNR_joint;
% Rec.PatchSize = PatchSize;
% Rec.olp = olp;
% Rec.y_hsi = y_hsi;
