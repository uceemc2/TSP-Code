
Row0 = 128;
Col0 = 128;
mea_HSI = 1;
PatchSize = 2; 
savename = 'Model_HSI_PS2_Ch33';
filename = 'scene7';
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
%scale = 4;
[Row, Col, Ch] = size(x_hsi0);
[Row, Col, rgb] = size(x_rgb0);

% Row0 = 256;
% Col0 = 256;
if(resize_flag)
for nc = 1:Ch
    x_hsi(:,:,nc) = imresize(x_hsi0(:,:,nc) , [Row0,Col0]);
end

for nc = 1:rgb
    x_rgb(:,:,nc) = imresize(x_rgb0(:,:,nc) , [Row0,Col0])/6;
end
end

[Row, Col, Ch] = size(x_hsi);

%load the trained GMM
load(savename);
PatchSize = sqrt(size(Mu_hsi,1)/Ch);

Row_new = floor(Row/PatchSize)*PatchSize;
Col_new = floor(Col/PatchSize)*PatchSize;
x_hsi = x_hsi(1:Row_new, 1:Col_new,:);
x_rgb = x_rgb(1:Row_new, 1:Col_new,:);


% generate the Phi_matrix with the CASSI scenario
T = mea_HSI;
for t= 1:T
Phi{t} = binornd(1,0.5,[Row_new, Col_new,Ch]);
end
shift = 1;
shift_T = 5;
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

% now y is the measurement of the HSI image
for t1 = 1:T 
    y_hsi{t1} = sum(x_hsi.*Phi_new{t1},3);
end

% now we get the measurement of the rgb image
if(strcmp(mea_rgb,'rgb'))
    y_rgb = x_rgb;
    Phi_rgb_p = sparse(eye(PatchSize^2*rgb));
end

olp = PatchSize; % here we use non-overlapping patches
% vectorize the patches
y_hsi_all = reshape(cell2mat(y_hsi),[Row_new, Col_new,T]);
y_hsi_p = video2patches_fast(y_hsi_all,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'
y_rgb_p = video2patches_fast(y_rgb,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'
x_hsi_p = video2patches_fast(x_hsi,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'
x_rgb_p = video2patches_fast(x_rgb,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'
% 
Phi_hsi = getw_unit(Phi_new_unit);

Phi_both = [Phi_hsi zeros(size(Phi_hsi,1),size(Phi_rgb_p,2));...
            zeros(size(Phi_rgb_p,1), size(Phi_hsi,2))  Phi_rgb_p];
        
        [M1 n1]=size(Phi_hsi)
        r1=rank(Phi_hsi)
        [M2 n2]=size(Phi_rgb_p)
        figure(10)
        imagesc(Phi_hsi)
        figure(11)
        imagesc(Phi_rgb_p)

X_rec_hsi = zeros(size(x_hsi_p));
X_rec_rgb = zeros(size(x_rgb_p));


X_rec_hsi = GMM_CS_Inv_samePhi(y_hsi_p,Phi_hsi,Sig_hsi,Mu_hsi,pai_hsi);

image_recon_hsi = patches2video_fast(X_rec_hsi, Row_new, Col_new,Ch, olp, olp);
%figure; imshow(image_recon_lar);
PSNR = SS_PSNR_3D(x_hsi,image_recon_hsi);

X_rec_both = GMM_CS_Inv_samePhi([y_hsi_p; y_rgb_p],Phi_both,Sig,Mu,pai);

image_recon_joint_hsi = patches2video_fast(X_rec_both(1:PatchSize^2*Ch,:), Row_new, Col_new,Ch, olp, olp);
%figure; imshow(image_recon_joint_lar);
PSNR_joint = SS_PSNR_3D(x_hsi,image_recon_joint_hsi);

Wvls = 410:10:710;
vects=6; vects2=5; mid = 3;
dispCubeAshwin(x_hsi,1,Wvls,[],vects,vects2,0,1,'truth')

dispCubeAshwin(image_recon_hsi,1,Wvls,[],vects,vects2,0,1,'wo_side')
dispCubeAshwin(image_recon_joint_hsi,1,Wvls,[],vects,vects2,0,1,'w_side')
% figure; subplot(121); imshow(image_recon_lar); title(['W/O side, PSNR: ' num2str(PSNR)]);
% subplot(122); imshow(image_recon_joint_lar); title(['W side, PSNR: ' num2str(PSNR_joint)]);


Rec.img_wo_side = image_recon_hsi; % the reconstructed image w/o side information
Rec.X_hsi = x_hsi; % the truth large iamge
Rec.X_rgb = x_rgb; % the truth large iamge
Rec.img_w_side = image_recon_joint_hsi;
Rec.PSNR_wo_side = PSNR;
Rec.PSNR_w_side = PSNR_joint;
Rec.PatchSize = PatchSize;
Rec.olp = olp;
Rec.y_hsi = y_hsi;