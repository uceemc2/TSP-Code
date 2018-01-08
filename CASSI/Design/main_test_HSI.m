close all
clear all
clc

% % Code used to Train coupled GMM using EM algorithm and do rconstruction 
% Xin Yuan, ECE, Duke University
% xin.yuan@duke.edu
% revision date: July 26, 2014
%=======================================================
Train_flag = 0;
resize_flag = 1; % This means we resize the image to be small for fast testing
Row =256; Col=256; % This is the size you want to resize, if you set resize_flag = 0, this does not matter
Row =128; Col=128; % This is the size you want to resize, if you set resize_flag = 0, this does not matter


if(Train_flag)
PatchSize = 2; % the patch size of the large image
% the file to train the GMM
Train_img_file = 'scene6'; 
% train the GMM
savename = Train_couple_GMM_HSI(Train_img_file, PatchSize,resize_flag,Row,Col);
else
    savename = 'Model_HSI_PS2_Ch33';
end


% test the reconstruction
Test_img_file = 'scene7';
mea_HSI = 1; % measurement images of the HSI image 
mea_rgb = 'rgb'; % measurement format of the rgb image; 'rgb' or 'mosaic'


Rec = Rec_couple_GMM_HSI(savename, Test_img_file, mea_HSI, mea_rgb,resize_flag,Row,Col);
wave = 400:10:720;

figure; plot(wave, Rec.PSNR_wo_side); hold on; plot(wave, Rec.PSNR_w_side,'r->');
legend(['PSNR without rgb, average: ' num2str(mean(Rec.PSNR_wo_side))],...
    ['PSNR with rgb, average: '  num2str(mean(Rec.PSNR_w_side))]);
xlabel('wavelength, Spectral Channel'); ylabel('Reconstruction PSNR');
save([Test_img_file '_mea' num2str(mea_HSI) '_' mea_rgb '_psize' num2str(Rec.PatchSize) '_olp' num2str(Rec.olp)],'Rec');
figure; 
for m = 1:mea_HSI
subplot(1,mea_HSI,m); imshow(Rec.y_hsi{m}./max(max(Rec.y_hsi{m})))
end
