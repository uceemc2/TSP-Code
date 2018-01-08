%TRAIN FOR 6 images
savename = 'TrainGMMs_expC/Model_TRAIN6IMAGES_RGB_512_HSI_256_K10_TEST_FRO_S1_Ch33.mat'

K1K2=10;
NormPhi = 0;
filename = 'scene1';
nrandom = 1;



load('tmpPhiHsi.mat');


mea_HSI = 1;
mea_rgb = 'rgb';
resize_flag = 1;
Row_hsi = 256;
Col_hsi = 256;
PatchSize = 4;

% Row_rgb = 256;
% Col_rgb = 256;
% PatchSize_lar = 4;

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
%scale = 4;
[Row, Col, Ch] = size(x_hsi0);
[Row, Col, rgb] = size(x_rgb0);

% Row0 = 256;
% Col0 = 256;
if(resize_flag)
for nc = 1:Ch
    x_hsi(:,:,nc) = imresize(x_hsi0(:,:,nc) , [Row_hsi,Col_hsi]);
end

for nc = 1:rgb
    x_rgb(:,:,nc) = imresize(x_rgb0(:,:,nc) , [Row_rgb,Col_rgb]);
end
end

% imwrite(x_rgb,'RGB_BeforeGamma.png','png');
% hgamma = ...
%    vision.GammaCorrector(2.0,'Correction','De-gamma');
% x_rgb = step(hgamma, x_rgb);

[Row, Col, Ch] = size(x_hsi);

%load the trained GMM
load(savename);
% PatchSize = sqrt(size(Mu_hsi,1)/Ch);


Row_new = floor(Row/PatchSize)*PatchSize;
Col_new = floor(Col/PatchSize)*PatchSize;
x_hsi = x_hsi(1:Row_new, 1:Col_new,:);


[Row, Col, rgb] = size(x_rgb);
Row_new_rgb = floor(Row/PatchSize_lar)*PatchSize_lar;
Col_new_rgb = floor(Col/PatchSize_lar)*PatchSize_lar;
x_rgb = x_rgb(1:Row_new_rgb, 1:Col_new_rgb,:);

% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
np=33; NumM2=linspace(1,33,np);
for mea_HSI=1:length(NumM2)
    
    kk = mea_HSI;
    
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
olp_lar=PatchSize_lar;
% vectorize the patches
y_hsi_all = reshape(cell2mat(y_hsi),[Row_new, Col_new,T]);
y_hsi_p = video2patches_fast(y_hsi_all,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'
% y_rgb_p = video2patches_fast(y_rgb,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'
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

str='';
str= sprintf('Tmp_PhiHsi_%d=Phi_hsi;',mea_HSI);
eval(str);
end