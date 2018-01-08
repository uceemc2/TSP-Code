nrandom = 5;
K1K2=30;
filename = 'scene4';
mea_HSI = 1;
mea_rgb = 'rgb';
resize_flag = 1;
Row_hsi = 256;
Col_hsi = 256;
Row_rgb = 256;
Col_rgb = 256;
PatchSize_lar = 4;


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
PatchSize = sqrt(size(Mu_hsi,1)/Ch);

Row_new = floor(Row/PatchSize)*PatchSize;
Col_new = floor(Col/PatchSize)*PatchSize;
x_hsi = x_hsi(1:Row_new, 1:Col_new,:);


[Row, Col, rgb] = size(x_rgb);
Row_new_rgb = floor(Row/PatchSize_lar)*PatchSize_lar;
Col_new_rgb = floor(Col/PatchSize_lar)*PatchSize_lar;
x_rgb = x_rgb(1:Row_new_rgb, 1:Col_new_rgb,:);

imwrite(x_rgb,'RGBoriginal.png','png');