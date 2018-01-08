
function savename = Train_couple_GMM(filename, PatchSize_lar,scale)
% Code used to Train GMM using EM algorithm
% Xin Yuan, ECE, Duke University
% xin.yuan@duke.edu
% revision date: March 14, 2013
%=======================================================

%clc; clear all; close all;

% Read image, you can use many different images to train the dictionary.

X0= imread(filename);
[Row, Col,rgb] = size(X0);
if(rgb >1)
X= im2double(rgb2gray(X0));
else
X= im2double(X0);
end
%figure; imagesc(X); colormap gray
%scale = 4;
[Row, Col] = size(X);

X_small = imresize(X,[Row/scale, Col/scale]);

%PatchSize_lar = 8; % Define the patchsize
PatchSize_sml = PatchSize_lar/scale; 
% vectorize the patches
image_lar = im2col(X,[PatchSize_lar PatchSize_lar], 'distinct');  % Here you can use 'sliding'
image_sml = im2col(X_small,[PatchSize_sml PatchSize_sml], 'distinct');  % Here you can use 'sliding'
%image = image(:,1:10000);
image = [image_lar; image_sml];

% Define the parameters of EM
options = statset('Display','iter','MaxIter',500);
K = 20;  % The GMM component number
obj = gmdistribution.fit(image.',K,'Regularize',10^-3,'Options',options);
pai = obj.PComponents; Mu = obj.mu'; Sig = obj.Sigma;

obj1 = gmdistribution.fit(image_lar.',K,'Regularize',10^-3,'Options',options);
pai_lar = obj1.PComponents; Mu_lar = obj1.mu'; Sig_lar = obj1.Sigma;

obj2 = gmdistribution.fit(image_sml.',K,'Regularize',10^-3,'Options',options);
pai_sml = obj2.PComponents; Mu_sml = obj2.mu'; Sig_sml = obj2.Sigma;

savename = ['Model_SR_PL' num2str(PatchSize_lar) '_PS' num2str(PatchSize_sml) '.mat'];
save(savename,...
    '-v7.3', 'Mu','Sig','pai','Mu_lar','Sig_lar','pai_lar','Mu_sml','Sig_sml','pai_sml');