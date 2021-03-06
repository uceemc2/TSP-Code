function savename = Train_couple_GMM_HSI(filename, PatchSize,resize_flag,Row0,Col0)
% Code used to Train GMM using EM algorithm
% Xin Yuan, ECE, Duke University
% xin.yuan@duke.edu
% revision date:July 26, 2014
%=======================================================

%clc; clear all; close all;
filename = 'scene3';
PatchSize = 4 %2
resize_flag = 1;
Row_hsi = 256;%128
Col_hsi = 256;%128
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

if(resize_flag)
for nc = 1:Ch
%     x_hsi(:,:,nc) = imresize(x_hsi0(:,:,nc) , [Row0,Col0]);
x_hsi(:,:,nc) = imresize(x_hsi0(:,:,nc) , [Row_hsi,Col_hsi]);
end

for nc = 1:rgb
%     x_rgb(:,:,nc) = imresize(x_rgb0(:,:,nc) , [Row0,Col0])/6;
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
image = [image_rgb; image_hsi];

% image_hsi = im2col(X,[PatchSize_lar PatchSize_lar], 'distinct');  % Here you can use 'sliding'
% image_sml = im2col(X_small,[PatchSize_sml PatchSize_sml], 'distinct');  % Here you can use 'sliding'
% image = [image_hsi; image_rgb];

% Define the parameters of EM
options = statset('Display','iter','MaxIter',200);
K = 20;  % The GMM component number
obj = gmdistribution.fit(image.',K,'Regularize',10^-3,'Options',options);
pai = obj.PComponents; Mu = obj.mu'; Sig = obj.Sigma;

obj2 = gmdistribution.fit(image_hsi.',K,'Regularize',10^-3,'Options',options);
pai_hsi = obj2.PComponents; Mu_hsi = obj2.mu'; Sig_hsi = obj2.Sigma;

obj1 = gmdistribution.fit(image_rgb.',K,'Regularize',10^-3,'Options',options);
pai_rgb = obj1.PComponents; Mu_rgb = obj1.mu'; Sig_rgb = obj1.Sigma;

savename = ['Model_HSI_PS_MYChen' num2str(PatchSize) '_Ch' num2str(Ch) '.mat'];
save(savename,...
    '-v7.3', 'Mu','Sig','pai',...
    'Mu_hsi','Sig_hsi','pai_hsi',...
    'Mu_rgb','Sig_rgb','pai_rgb');