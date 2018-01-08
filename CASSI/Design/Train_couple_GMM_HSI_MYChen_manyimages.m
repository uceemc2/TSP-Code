function savename = Train_couple_GMM_HSI(filename, PatchSize,resize_flag,Row0,Col0)
% Code used to Train GMM using EM algorithm
% Xin Yuan, ECE, Duke University
% xin.yuan@duke.edu
% revision date:July 26, 2014
%=======================================================
%clc; clear all; close all;
% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx image 1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
filename = 'scene1';
PatchSize = 4 %2
resize_flag = 1;
Row_hsi = 256;%128
Col_hsi = 256;%128
Row_rgb = 256;
Col_rgb = 256;
PatchSize_lar = 4;
% Row_rgb = 512;
% Col_rgb = 512;
% PatchSize_lar = 8;
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

if(resize_flag)
for nc = 1:Ch
x_hsi(:,:,nc) = imresize(x_hsi0(:,:,nc) , [Row_hsi,Col_hsi]);
end

for nc = 1:rgb
x_rgb(:,:,nc) = imresize(x_rgb0(:,:,nc) , [Row_rgb,Col_rgb]);
end
end

[Row, Col, Ch] = size(x_hsi);
Row_new = floor(Row/PatchSize)*PatchSize;
Col_new = floor(Col/PatchSize)*PatchSize;
x_hsi = x_hsi(1:Row_new, 1:Col_new,:);

[Row, Col, rgb] = size(x_rgb);
Row_new = floor(Row/PatchSize_lar)*PatchSize_lar;
Col_new = floor(Col/PatchSize_lar)*PatchSize_lar;
x_rgb = x_rgb(1:Row_new, 1:Col_new,:);

olp = PatchSize; % here we use non-overlapping patches
olp_lar = PatchSize_lar;
% vectorize the patches
image_rgb = video2patches_fast(x_rgb,PatchSize_lar, PatchSize_lar,olp_lar,olp_lar);  % Here you can use 'sliding'
image_hsi = video2patches_fast(x_hsi,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'
% image = [image_rgb; image_hsi];
image_rgb_1 = image_rgb;
image_hsi_1 = image_hsi;

% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx image 2 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
filename = 'scene2';
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

if(resize_flag)
for nc = 1:Ch
x_hsi(:,:,nc) = imresize(x_hsi0(:,:,nc) , [Row_hsi,Col_hsi]);
end

for nc = 1:rgb
x_rgb(:,:,nc) = imresize(x_rgb0(:,:,nc) , [Row_rgb,Col_rgb]);
end
end

[Row, Col, Ch] = size(x_hsi);
Row_new = floor(Row/PatchSize)*PatchSize;
Col_new = floor(Col/PatchSize)*PatchSize;
x_hsi = x_hsi(1:Row_new, 1:Col_new,:);

[Row, Col, rgb] = size(x_rgb);
Row_new = floor(Row/PatchSize_lar)*PatchSize_lar;
Col_new = floor(Col/PatchSize_lar)*PatchSize_lar;
x_rgb = x_rgb(1:Row_new, 1:Col_new,:);

olp = PatchSize; % here we use non-overlapping patches
olp_lar = PatchSize_lar;
% vectorize the patches
image_rgb = video2patches_fast(x_rgb,PatchSize_lar, PatchSize_lar,olp_lar,olp_lar);  % Here you can use 'sliding'
image_hsi = video2patches_fast(x_hsi,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'

image_rgb_2 = image_rgb;
image_hsi_2 = image_hsi;
% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx image 3 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
filename = 'scene3';
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

if(resize_flag)
for nc = 1:Ch
x_hsi(:,:,nc) = imresize(x_hsi0(:,:,nc) , [Row_hsi,Col_hsi]);
end

for nc = 1:rgb
x_rgb(:,:,nc) = imresize(x_rgb0(:,:,nc) , [Row_rgb,Col_rgb]);
end
end

[Row, Col, Ch] = size(x_hsi);
Row_new = floor(Row/PatchSize)*PatchSize;
Col_new = floor(Col/PatchSize)*PatchSize;
x_hsi = x_hsi(1:Row_new, 1:Col_new,:);

[Row, Col, rgb] = size(x_rgb);
Row_new = floor(Row/PatchSize_lar)*PatchSize_lar;
Col_new = floor(Col/PatchSize_lar)*PatchSize_lar;
x_rgb = x_rgb(1:Row_new, 1:Col_new,:);

olp = PatchSize; % here we use non-overlapping patches
olp_lar = PatchSize_lar;
% vectorize the patches
image_rgb = video2patches_fast(x_rgb,PatchSize_lar, PatchSize_lar,olp_lar,olp_lar);  % Here you can use 'sliding'
image_hsi = video2patches_fast(x_hsi,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'

image_rgb_3 = image_rgb;
image_hsi_3 = image_hsi;
% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx image 4 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
filename = 'scene4';
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

if(resize_flag)
for nc = 1:Ch
x_hsi(:,:,nc) = imresize(x_hsi0(:,:,nc) , [Row_hsi,Col_hsi]);
end

for nc = 1:rgb
x_rgb(:,:,nc) = imresize(x_rgb0(:,:,nc) , [Row_rgb,Col_rgb]);
end
end

[Row, Col, Ch] = size(x_hsi);
Row_new = floor(Row/PatchSize)*PatchSize;
Col_new = floor(Col/PatchSize)*PatchSize;
x_hsi = x_hsi(1:Row_new, 1:Col_new,:);

[Row, Col, rgb] = size(x_rgb);
Row_new = floor(Row/PatchSize_lar)*PatchSize_lar;
Col_new = floor(Col/PatchSize_lar)*PatchSize_lar;
x_rgb = x_rgb(1:Row_new, 1:Col_new,:);

olp = PatchSize; % here we use non-overlapping patches
olp_lar = PatchSize_lar;
% vectorize the patches
image_rgb = video2patches_fast(x_rgb,PatchSize_lar, PatchSize_lar,olp_lar,olp_lar);  % Here you can use 'sliding'
image_hsi = video2patches_fast(x_hsi,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'

image_rgb_4 = image_rgb;
image_hsi_4 = image_hsi;

% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx image 6 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
filename = 'scene6';
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

if(resize_flag)
for nc = 1:Ch
x_hsi(:,:,nc) = imresize(x_hsi0(:,:,nc) , [Row_hsi,Col_hsi]);
end

for nc = 1:rgb
x_rgb(:,:,nc) = imresize(x_rgb0(:,:,nc) , [Row_rgb,Col_rgb]);
end
end

[Row, Col, Ch] = size(x_hsi);
Row_new = floor(Row/PatchSize)*PatchSize;
Col_new = floor(Col/PatchSize)*PatchSize;
x_hsi = x_hsi(1:Row_new, 1:Col_new,:);

[Row, Col, rgb] = size(x_rgb);
Row_new = floor(Row/PatchSize_lar)*PatchSize_lar;
Col_new = floor(Col/PatchSize_lar)*PatchSize_lar;
x_rgb = x_rgb(1:Row_new, 1:Col_new,:);

olp = PatchSize; % here we use non-overlapping patches
olp_lar = PatchSize_lar;
% vectorize the patches
image_rgb = video2patches_fast(x_rgb,PatchSize_lar, PatchSize_lar,olp_lar,olp_lar);  % Here you can use 'sliding'
image_hsi = video2patches_fast(x_hsi,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'

image_rgb_6 = image_rgb;
image_hsi_6 = image_hsi;
% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx image 7 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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

if(resize_flag)
for nc = 1:Ch
x_hsi(:,:,nc) = imresize(x_hsi0(:,:,nc) , [Row_hsi,Col_hsi]);
end

for nc = 1:rgb
x_rgb(:,:,nc) = imresize(x_rgb0(:,:,nc) , [Row_rgb,Col_rgb]);
end
end

[Row, Col, Ch] = size(x_hsi);
Row_new = floor(Row/PatchSize)*PatchSize;
Col_new = floor(Col/PatchSize)*PatchSize;
x_hsi = x_hsi(1:Row_new, 1:Col_new,:);

[Row, Col, rgb] = size(x_rgb);
Row_new = floor(Row/PatchSize_lar)*PatchSize_lar;
Col_new = floor(Col/PatchSize_lar)*PatchSize_lar;
x_rgb = x_rgb(1:Row_new, 1:Col_new,:);

olp = PatchSize; % here we use non-overlapping patches
olp_lar = PatchSize_lar;
% vectorize the patches
image_rgb = video2patches_fast(x_rgb,PatchSize_lar, PatchSize_lar,olp_lar,olp_lar);  % Here you can use 'sliding'
image_hsi = video2patches_fast(x_hsi,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'

image_rgb_7 = image_rgb;
image_hsi_7 = image_hsi;

% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx image 8 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
filename = 'scene8';
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

if(resize_flag)
for nc = 1:Ch
x_hsi(:,:,nc) = imresize(x_hsi0(:,:,nc) , [Row_hsi,Col_hsi]);
end

for nc = 1:rgb
x_rgb(:,:,nc) = imresize(x_rgb0(:,:,nc) , [Row_rgb,Col_rgb]);
end
end

[Row, Col, Ch] = size(x_hsi);
Row_new = floor(Row/PatchSize)*PatchSize;
Col_new = floor(Col/PatchSize)*PatchSize;
x_hsi = x_hsi(1:Row_new, 1:Col_new,:);

[Row, Col, rgb] = size(x_rgb);
Row_new = floor(Row/PatchSize_lar)*PatchSize_lar;
Col_new = floor(Col/PatchSize_lar)*PatchSize_lar;
x_rgb = x_rgb(1:Row_new, 1:Col_new,:);

olp = PatchSize; % here we use non-overlapping patches
olp_lar = PatchSize_lar;
% vectorize the patches
image_rgb = video2patches_fast(x_rgb,PatchSize_lar, PatchSize_lar,olp_lar,olp_lar);  % Here you can use 'sliding'
image_hsi = video2patches_fast(x_hsi,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'

image_rgb_8 = image_rgb;
image_hsi_8 = image_hsi;

% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx image new xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
image_rgb_new  = [ image_rgb_1 image_rgb_2  image_rgb_3 image_rgb_4 image_rgb_6 image_rgb_7 image_rgb_8];
image_hsi_new  = [ image_hsi_1 image_hsi_2  image_hsi_3 image_hsi_4 image_hsi_6 image_hsi_7 image_hsi_8];
% image_rgb_new  = [ image_rgb_2  image_rgb_3 image_rgb_4 image_rgb_6 image_rgb_7 image_rgb_8];
% image_hsi_new  = [ image_hsi_2  image_hsi_3 image_hsi_4 image_hsi_6 image_hsi_7 image_hsi_8];
% image_rgb_new  = [ image_rgb_3];
% image_hsi_new  = [ image_hsi_3];

image = [image_rgb_new; image_hsi_new];
% Define the parameters of EM
options = statset('Display','iter','MaxIter',500);

K = 30;  % The GMM component number
obj = gmdistribution.fit(image.',K,'Regularize',10^-3,'Options',options);
pai = obj.PComponents; Mu = obj.mu'; Sig = obj.Sigma;

obj2 = gmdistribution.fit(image_hsi_new.',K,'Regularize',10^-3,'Options',options);
pai_hsi = obj2.PComponents; Mu_hsi = obj2.mu'; Sig_hsi = obj2.Sigma;

obj1 = gmdistribution.fit(image_rgb_new.',K,'Regularize',10^-3,'Options',options);
pai_rgb = obj1.PComponents; Mu_rgb = obj1.mu'; Sig_rgb = obj1.Sigma;

savename = 'TrainGMMs_expC/Model_ALLTRAIN_RGB_256_HSI_256_K30_TEST_FRO_ALL_Ch33_1111.mat';
% savename = 'TrainGMMs_expC/Model_ALLTRAIN_RGB_512_HSI_256_K30_TEST_FRO_ALL_Ch33.mat';
% savename = 'TrainGMMs_expC/Model_TRAIN6IMAGES_RGB_512_HSI_256_K30_TEST_FRO_S1_Ch33.mat';



% savename = ['Model_RGB_256_HSI_256_K10_TRAIN_ONLY_S3_Ch' num2str(Ch) '.mat'];
% savename = ['Model_RGB_256_HSI_256_K10_TEST_FRO_ALL_' num2str(PatchSize) '_Ch' num2str(Ch) '.mat'];

save(savename,...
    '-v7.3', 'Mu','Sig','pai',...
    'Mu_hsi','Sig_hsi','pai_hsi',...
    'Mu_rgb','Sig_rgb','pai_rgb');