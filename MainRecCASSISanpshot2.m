addpath('./CASSI/Design');
savename = 'Model_TRAIN6IMAGES_RGB_512_HSI_256_K20_TEST_FRO_S1_Ch33.mat';
K1K2=20;
NormPhi = 1;
filename = './CASSI/Design/scene1';
nrandom = 1; 

% ************************
olp_lar = 4;
olp = 2; 
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

for mea_HSI=1:2 
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

% Load gsvd-based matrix
load('tmpPhiGSVD_1_M32K20.mat'); 
re_Phi_hsi = new_Phi_hsi;
re_Phi_hsi_1  = [];
re_Phi_hsi_2  = [];
for i_s = 1:2
 
    if i_s == 1
        re_Phi_hsi_tmp =  re_Phi_hsi(1:16,:);
    else
        re_Phi_hsi_tmp =  re_Phi_hsi(17:end,:);
    end
    
    for j=1:33
        diag_gsvd(:,j) = diag(re_Phi_hsi_tmp(:,(j*16-15):j*16));
    end
    [d_row d_col]=size(diag_gsvd);

    DesCodedMask1 = [];
    DesCodedMask2 = [];
    DesCodedMask3 = [];
    for  i=1:4
        for j=1:3
            idx1 = 0;
            idx2 = 0;
            idx3 = 0;

            distmp1 = 0;
            distmp0 = 0;
        
            if j == 1
                idx1 = diag_gsvd(1+4*(i-1),j+30);
                idx2 = diag_gsvd(2+4*(i-1),j+31);
                idx3 = diag_gsvd(3+4*(i-1),j+32);  

                distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2)+norm(abs(1-idx3),2);
                distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2)+norm(abs(idx3),2);
            elseif j == 2
                idx1 = diag_gsvd(1+4*(i-1),j+30);
                idx2 = diag_gsvd(2+4*(i-1),j+31);  
           
                distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2);
                distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2);
            elseif j == 3
                idx1 = diag_gsvd(1+4*(i-1),j+30);
                distmp1 = norm(abs(1-idx1),2);
                distmp0 = norm(abs(idx1),2);           
            end                
        
            if distmp1 > distmp0
                DesCodedMask1(j,i)=0;
            else
                DesCodedMask1(j,i)=1;
            end
        end
    end
    DesCodedMask1 = flipud(DesCodedMask1);

    for  i=1:4
        for j=1:30
            idx1 = 0;
            idx2 = 0;
            idx3 = 0;
            idx4 = 0;
        
            idx1 = diag_gsvd(1+4*(i-1),j);
            idx2 = diag_gsvd(2+4*(i-1),j+1);
            idx3 = diag_gsvd(3+4*(i-1),j+2);  
            idx4 = diag_gsvd(4+4*(i-1),j+3);
        
            distmp1 = 0;
            distmp0 = 0;
        
            distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2)+norm(abs(1-idx3),2)+norm(abs(1-idx4),2);
            distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2)+norm(abs(idx3),2)+norm(abs(idx4),2);
        
            if distmp1 > distmp0
                DesCodedMask2(j,i)=0;
            else
                DesCodedMask2(j,i)=1;
            end
        end
    end
    DesCodedMask2 = flipud(DesCodedMask2);

    for  i=1:4
        for j=1:3
            idx1 = 0;
            idx2 = 0;
            idx3 = 0;
            distmp1 = 0;
            distmp0 = 0;
        
            if j == 1
                idx1 = diag_gsvd(2+4*(i-1),j);
                idx2 = diag_gsvd(3+4*(i-1),j+1);
                idx3 = diag_gsvd(4+4*(i-1),j+2);
                distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2)+norm(abs(1-idx3),2);
                distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2)+norm(abs(idx3),2);
            elseif j == 2
                idx1 = diag_gsvd(3+4*(i-1),j-1);
                idx2 = diag_gsvd(4+4*(i-1),j);
                distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2);
                distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2);
            else
                idx1 = diag_gsvd(4+4*(i-1),j-2); 
                distmp1 = norm(abs(1-idx1),2);
                distmp0 = norm(abs(idx1),2);
            end
        
            if distmp1 > distmp0
                DesCodedMask3(j,i)=0;
            else
                DesCodedMask3(j,i)=1;
            end        
        end
    end
    DesCodedMask = [DesCodedMask1 ; DesCodedMask2 ; DesCodedMask3];

%     DesCodedMask

    RescalePhi2=[];
    for i=1:33
        tempPhi2(:,:,i) = DesCodedMask(33-i+1:36-i+1,:);
        vtempPhi2 = reshape(tempPhi2(:,:,i),16,[]);
        RescalePhi2 = [RescalePhi2 diag(vtempPhi2) ];
    end
    i_s
    if i_s == 1
       re_Phi_hsi_1 =  RescalePhi2;
    end
    if i_s == 2
       re_Phi_hsi_2 =  RescalePhi2;
    end
end
DesignPhi2 = [re_Phi_hsi_1;re_Phi_hsi_2];

% load('Overlapping_PhiHsi.mat');
%****************************************************
% Fixed measurements kernel by function CASSIdesign()
%   FinalPhi2 = CASSIdesign();
%****************************************************
    load('RandomPhi2_32.mat');
    load('FinalPhi2_32_02.mat');
    for nr=1:nrandom
        
        % Ramdom CASSI measurements
        Phi_both = [Phi_rgb_p zeros(size(Phi_rgb_p,1),size(Phi_hsi,2));...
            zeros(size(Phi_hsi,1), size(Phi_rgb_p,2))  Phi_hsi];
        
        [M2 n2]=size(Phi_hsi)
        r2=rank(Phi_hsi)
        [M1 n1]=size(Phi_rgb_p)
        
        X_rec_hsi = zeros(size(x_hsi_p));
        X_rec_rgb = zeros(size(x_rgb_p));
        y_rgb_p = Phi_rgb_p*x_rgb_p;

        y_hsi_p = Phi_hsi*x_hsi_p;
        X_rec_both = GMM_CS_Inv_samePhi([y_rgb_p;y_hsi_p],Phi_both,Sig,Mu,pai);        

        image_recon_joint_RGB = patches2video_fast(X_rec_both(1:PatchSize_lar^2*rgb,:), Row_rgb, Col_rgb,rgb, olp_lar, olp_lar);
        RECNAME ='';
        RECNAME = sprintf('CASSI_random_2snapshots.png',kk);
        imwrite(image_recon_joint_RGB,RECNAME,'png');

        PSNR_joint_RGB_random(nr,kk)  = SS_PSNR_3D(x_rgb_p,X_rec_both(1:PatchSize_lar^2*rgb,:));
        PSNR_joint_RGB_random
        
        % Designed CASSI measurements
        Phi_both = [Phi_rgb_p zeros(size(Phi_rgb_p,1),size(FinalPhi2,2));...
            zeros(size(FinalPhi2,1), size(Phi_rgb_p,2))  FinalPhi2];
        
        [M2 n2]=size(FinalPhi2)
        r2=rank(FinalPhi2)
        [M1 n1]=size(Phi_rgb_p)    
        
        X_rec_hsi = zeros(size(x_hsi_p));
        X_rec_rgb = zeros(size(x_rgb_p));
        y_rgb_p = Phi_rgb_p*x_rgb_p;

        y_hsi_p = FinalPhi2*x_hsi_p;
        X_rec_both = GMM_CS_Inv_samePhi([y_rgb_p;y_hsi_p],Phi_both,Sig,Mu,pai);    
        
        image_recon_joint_RGB = patches2video_fast(X_rec_both(1:PatchSize_lar^2*rgb,:), Row_rgb, Col_rgb,rgb, olp_lar, olp_lar);
        RECNAME ='';
        RECNAME = sprintf('CASSI_design_2snapshots.png',kk);
        imwrite(image_recon_joint_RGB,RECNAME,'png');  
        
        PSNR_joint_RGB_design(nr,kk)  = SS_PSNR_3D(x_rgb_p,X_rec_both(1:PatchSize_lar^2*rgb,:));
        PSNR_joint_RGB_design
    end
end
imwrite(x_rgb,'RGBoriginal.png','png');
