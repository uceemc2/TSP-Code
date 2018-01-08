% np=33; vettSNRdB=linspace(1,33,np);
np=16; vettSNRdB=linspace(1,16,np);
% np=32; vettSNRdB=linspace(16,512,np);
% np=16; vettSNRdB=linspace(16,256,np);
% vettSNRdB = [4 8 16 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256];



% PSNRrandom_K10(1,:) = 32.0627
% PSNRrandom_K10(2,:) = 31.8490
% PSNRrandom_K10(3,:) = 32.1771
% PSNRrandom_K10(4,:) = 33.2542
% PSNRrandom_K10(5,:) = 34.0610
% PSNRrandom_K10(6,:) = 33.7389
% PSNRrandom_K10(7,:) = 33.8320
% PSNRrandom_K10(8,:) = 33.5190
% PSNRrandom_K10(9,:) = 34.3383
% PSNRrandom_K10(10,:) = 34.3963

% PSNRrandom_K20(1,:) = 31.6314
% PSNRrandom_K20(2,:) = 30.1809
% PSNRrandom_K20(3,:) = 30.6528
% PSNRrandom_K20(4,:) = 31.2402
% PSNRrandom_K20(5,:) = 31.4125
% PSNRrandom_K20(6,:) = 30.6392
% PSNRrandom_K20(7,:) = 30.5107
% PSNRrandom_K20(8,:) = 30.7695
% PSNRrandom_K20(9,:) = 30.6975
% PSNRrandom_K20(10,:) = 30.5398

% PSNRrandom_K30(1,:) = 31.2658
% PSNRrandom_K30(2,:) = 31.0200
% PSNRrandom_K30(3,:) = 31.4596
% PSNRrandom_K30(4,:) = 31.5096
% PSNRrandom_K30(5,:) = 31.6087
% PSNRrandom_K30(6,:) = 31.2536
% PSNRrandom_K30(7,:) = 31.5549
% PSNRrandom_K30(8,:) = 31.3088
% PSNRrandom_K30(9,:) = 31.2581
% PSNRrandom_K30(10,:) = 31.5661


figure
plot(vettSNRdB',PSNR_joint_RGB_10.','-','LineWidth',1)
xlabel('CASSI Measurements ratio')
% xlabel('Gaussian Measurements m_2')
ylabel('PSNR (dB)')
grid on
hold on
% plot(vettSNRdB',PSNR_joint_RGB_20.','-o','LineWidth',1)
% plot(vettSNRdB',PSNR_joint_RGB_30.','-o','LineWidth',1)


plot(vettSNRdB',PSNR_joint_RGB_10_CASSIopt.','->','LineWidth',1) 
% plot(vettSNRdB',PSNR_joint_RGB_20_CASSIopt.','->','LineWidth',1) 
% plot(vettSNRdB',PSNR_joint_RGB_30_CASSIopt.','->','LineWidth',1) 


plot(vettSNRdB',PSNR_joint_RGB_10_test.','-o','LineWidth',1) 

% plot(vettSNRdB',PSNR_joint_RGB_10_opt.','->','LineWidth',1)
% plot(vettSNRdB',PSNR_joint_RGB_20_opt.','->','LineWidth',1)
% plot(vettSNRdB',PSNR_joint_RGB_30_opt.','->','LineWidth',1)