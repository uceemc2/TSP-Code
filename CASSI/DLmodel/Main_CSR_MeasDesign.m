% =========================================================================
% 
% Multimodal Image Super-resolution via Joint Sparse Representations 
% induced by Coupled Dictionaries 
% 
% 31/08/2017 Super-resolving RGB images with NIR images as guidance information.
% The measurement process is not blurring + downsampling. Instead, each measurement is a predifined linear combination of an image patch. The linear combination setting is included in the measurement matrix design from Mengyang et al.'s GMM work.
% 
% 
% This code is to solve the following Coupled Super-Resolution problem: 
% Step 1: Obtain the sparse representations Z, U, V from XL and Y;
% min_{Z, U, V} || [XL ; Y] - [Psi_cLR; Phi_c] * Z - [Psi_LR, 0; 0, Phi] * [U; V] ||_F^2;
% s.t. ||z_i||_0 <= s_c; ||u_i||_0 <= s_x; ||v_i||_0 <= s_y; \forall i
% Step 2: Reconstruct the HR image X;
% X = Psi_c * Z + Psi * U;

% Please refer to
% ------------
% P. Song, X. Deng, J. F. Mota, N. Deligiannis, P. L. Dragotti, M. R. Rodrigues, 
% "Multimodal Image Super-resolution via Joint Sparse Representations 
% induced by Coupled Dictionaries", IEEE Trans. Image Process., 2017.
% 
% P. Song, J. F. Mota, N. Deligiannis, and M. R. Rodrigues, ¡°Coupled
% dictionary learning for multimodal image super-resolution,¡± in IEEE
% Global Conf. Signal Inform. Process. IEEE, 2016, pp. 162¨C166.
% =========================================================================
%%
addpath('../ksvd');
addpath('../ksvd/ksvdbox');
addpath('../ksvd/ksvdbox/private_ccode');
addpath('../ksvd/ompbox');
% addpath('../TestImages_Multispectral');
% addpath('../TrainImages_Multispectral');
addpath('./Dicts') ;
addpath('./utils')
addpath('./utils2')
 
% addpath('../spams-matlab/test_release');
% addpath('../spams-matlab/src_release');
% addpath('../spams-matlab/build');

clear;  close all;

% 	Random seed number
	seed = RandStream('mcg16807','Seed',31415);
	RandStream.setGlobalStream(seed);	
% 	seed = RandStream.getGlobalStream;

CDLname = 'CDL_CKSVD_D192x512_sc10sx2sy2_Iter100_T96774_Scale4_Date15-Sep-2017.mat';
% CDLname = 'training\CDL_CKSVD_D192x512_sc10sx2sy2_Iter100_T96774_Scale4_Date04-Sep-2017_2imgs_atom512_001.mat';
% CDLname = 'training\CDL_CKSVD_D192x512_sc10sx2sy2_Iter100_T96774_Scale4_Date04-Sep-2017_1img_atom512.mat'; 
% CDLname = 'training\CDL_CKSVD_D192x1024_sc10sx2sy2_Iter100_T384054_Scale4_Date04-Sep-2017_1img_atom1024.mat'; 
% CDLname = 'training\CDL_CKSVD_D192x1024_sc10sx2sy2_Iter100_T384054_Scale4_Date04-Sep-2017_2imgs_atom1024.mat';
% CDLname = 'training\CDL_CKSVD_D192x512_sc10sx2sy2_Iter100_T384054_Scale4_Date13-Sep-2017.mat';

load( CDLname );

Method_cell = {'srOMP', 'srOMP2'};
Method_RecoverStage = Method_cell{1}; 

% SR parameters
K = paramsCDL.K; % The number of atoms.
N = paramsCDL.NXL; % The length of one atom for LR dicts.
NX = paramsCDL.NXH;  % The length of one atom for HR dicts
NY = paramsCDL.NY; % The length of one atom for side info dicts

upscale = paramsCDL.upscale  ; 
variance_Thresh_SR = 0; % 0.02; % threshold for smooth patches


% column and row positions of four small dictionaries in the whole
% dictionary. as the whole dictionary = 
% 	[Psi_cx,   Psi_x,            zeros(N,K) ; ...
% 	Psi_cy,    zeros(N,K),    Psi_y];
ind1 = 1: K;
ind2 = (K+1) : 2*K;
ind3 = (2*K +1) : 3*K;

ind1_row = 1: N;
ind2_row = (N + 1) : 2*N;

% Trained dictionaries
Psi_cxLR = outputCDL.Psi_cxLR;
Psi_xLR = outputCDL.Psi_xLR;
Psi_cx = outputCDL.Psi_cx;
Psi_x = outputCDL.Psi_x;
Psi_cy = outputCDL.Psi_cy;
Psi_y = outputCDL.Psi_y;

% location, width used to crop image.
croploc = [1,1];
cropwidth = [512,512];
blocksize = [sqrt(N),sqrt(N)];  % the size of each image patch.
stepsize = [1,1];  % the step of extracting image patches. If stepsize < stepsize, there exists overlapping aeras among adjacent patches.
s_c = 30;
s_x = 2;
s_y = 2;	
Ssum_rec = s_c+s_x+s_y;  % threshold for OMP
lambda = 1e-5; % parameter for ridge regression
Row_rgb = 512; 
Col_rgb = 512;

% % fast test
% croploc = [100, 100];
% cropwidth = [512/2,512/2];
% stepsize = [4, 4]; 
% s_c = 4;
% s_x = 2;
% s_y = 2;
% Ssum_rec = s_c+s_x+s_y;  % threshold for OMP

paramsCSR.N = N;
paramsCSR.K = K;	
paramsCSR.cropwidth = cropwidth;
paramsCSR.croploc =croploc;
paramsCSR.blocksize = blocksize;
paramsCSR.stepsize = stepsize;
paramsCSR.s_c = s_c;
paramsCSR.s_x = s_x;
paramsCSR.s_y = s_y;
paramsCSR.Ssum_rec = Ssum_rec;
paramsCSR.upscale = upscale;
paramsCSR.lambda = lambda;
paramsCSR.variance_Thresh_SR = variance_Thresh_SR;
% --------------------------------------------------------
% Construct testing dataset.
% % Use multispectral images (with .png) as the images of interest . 
% directory = '../TestImages_Multispectral'; 
% patternX = '*.png';
% patternY = '*.bmp';
% 
% XpathCell = glob(directory, patternX );
% Xcell = load_images( XpathCell );
% 
% YpathCell = glob(directory, patternY );
% Ycell = load_images(YpathCell );
% 
% if length(Xcell) ~= length(Ycell)	
% 	error('Error: The number of X images is not equal to the number of Y images!');
% end
% 
% % crop a little of each image
% for i = 1: length(Xcell)
% 	Xcell(i) = modcrop(Xcell(i), upscale);
% 	Ycell(i) = modcrop(Ycell(i), upscale);
% end


for img_index = 1
% for img_index = 1: length(Xcell)

	disp('--------------------------------------------------------')
% 	fprintf('Processing image %d of total %d ... \n', img_index, length(Xcell));
% 	
% 	X_test = Xcell{img_index}; % testing image X
% 	Y_test = Ycell{img_index}; % testing image Y
% 	
% 	% obtain low resolution version.
% 	X_testLR = imresize(X_test, 1/upscale, 'bicubic');
% 	X_testLR = imresize(X_testLR, upscale, 'bicubic');	
% 	
% 	cropwidth = size(X_test);
% 	paramsCSR.cropwidth = cropwidth;
% 	
% 	X_test_vec = []; % vectorized patches from testing image X;
% 	Y_test_vec = []; % vectorized patches from testing image Y;
% 	X_test_vecLR = [];
% 	
% 	% add high-pass filtering options
% 	paramsCSR.Xcell = {X_test};
% 	paramsCSR.Ycell = {Y_test};
% 	paramsCSR.filter_flag = 0 ; % perform high-pass filtering on the LR images.
% 	paramsCSR.variance_Thresh = 0; % Discard those patch pairs with too small variance.
% 	
% 	% Load training image and construct training dataset.
% 	% Use RGB images with .bmp as the signal of interest and another wavelength as the SI. 
% 	
% 	[X_test_vec, X_test_vecLR, Y_test_vec, DC] = Sample_PreProcess_TestImgs( paramsCSR);
% 	dc_Y = DC.Yh;
% 	dc_XLR = DC.Xl;
% 	% remove data field to reduce the storage ccost.
% 	paramsCSR = rmfield(paramsCSR,'Xcell'); 
% 	paramsCSR = rmfield(paramsCSR,'Ycell');
	
% 	T = size(X_test_vec, 2);
% 	AtomLength = blocksize(1) * blocksize(2); % the length of atoms and vectorized patch. 8x8=64;




	X_test_vec = []; X_test_vecLR = []; Y_test_vec = [];
	TestImgNo = [1];
	for i = 1: length(TestImgNo)
		ImgName = ['scene', num2str(TestImgNo(i))];
		[P1, xh, xl, P2, yh, yl] = DLmodeltraining(ImgName); 
		yl=0.5*yl;
		% ground truth
		list_rgb = dir([ImgName '/*.bmp']);
		x_rgb0 = im2double(imread(list_rgb.name));
		for nc = 1:3
			X_test(:,:,nc) = imresize(x_rgb0(:,:,nc) , [Row_rgb,Col_rgb]);
		end

		X_test_vec = [X_test_vec, xh];
		X_test_vecLR = [X_test_vecLR, xl];
		Y_test_vec = [Y_test_vec, yl];
	end

	
	T = size(X_test_vec, 2);
	
	
	
	info = sprintf('Image Recovery with SI.');
	info = sprintf('%s \n Method with SI: %s. ', info, Method_RecoverStage);
	info = sprintf('%s Testing Data Size = %d x %d. ', info, N, T);
	info = sprintf('%s Ssum_rec = %d. ', info, Ssum_rec);
	disp(info);

	for i_MeasNo = 1 
		MeasNo = i_MeasNo * blocksize(1) * blocksize(2) / (upscale^2);

		disp(['Measurements = ', num2str(MeasNo)]);

		X_rec = zeros(NX, T); % recovered X;
		Y_rec = zeros(NY, T); % recovered Y;
		Z_cfound = zeros(size(Psi_cx,2), T); %
		Z_xfound = zeros(size(Psi_x,2), T);   
		Z_yfound = zeros(size(Psi_y,2), T);

% 		% store away those patches with small variance.
% 		X_test_vecLR0 = X_test_vecLR; 
% 		Y_test_vec0 = Y_test_vec;
% 		indCol = find(sum(X_test_vecLR.^2, 1) >= variance_Thresh_SR) ;
% 		X_test_vecLR = X_test_vecLR(:, indCol);
% 		Y_test_vec = Y_test_vec(:, indCol) ;

	% --------------------------------------------------------
		%% Recover With Side Information
		tic; % measure time
		switch Method_RecoverStage
			case 'srOMP'  % Enforcing more common components to take better advantage of guidance information.
			% use OMP package, but the sparse threshold can not be larger than measurements number
				data = [X_test_vecLR; Y_test_vec] ; 
				D = [Psi_cxLR; Psi_cy];
				D = D./ repmat(sqrt(sum(D.^2)) , [size(D ,1), 1]) ; 
				Z_cfound = omp(D' * data, D' * D, s_c); % fast		
% 				PP = inv(D' * D + lambda*eye(size(D,2))) * D' ; Z_cfound = PP * data;	% Use ridge regresson instead of OMP.
				
				X_lowR = X_test_vecLR - Psi_cxLR*Z_cfound;
				Y_highR = Y_test_vec - Psi_cy*Z_cfound;
				Z_xfound = omp(Psi_xLR' * X_lowR, Psi_xLR' *Psi_xLR, s_x);  % fast omp
% 				Z_xfound = omp(Psi_xLR, X_lowR, Psi_xLR' *Psi_xLR, s_x);  %	slow omp

				if sum( sum(Psi_y.^2) == 0) 
                    colZero = find(sum(Psi_y.^2) == 0); 
                    Psi_y(:, colZero) = sqrt(1/size(Psi_y,1)); 
                    Psi_y = Psi_y./ repmat(sqrt(sum(Psi_y.^2)) , [size(Psi_y ,1), 1]) ;  
				end
				Z_yfound = omp(Psi_y' * Y_highR, Psi_y'*Psi_y, s_y); 
				
			case 'srOMP2'
				data = [X_test_vecLR; Y_test_vec] ; 
				if sum( sum(Psi_y.^2) == 0) 
                    colZero = find(sum(Psi_y.^2) == 0); 
                    Psi_y(:, colZero) = sqrt(1/size(Psi_y,1)); 
                    Psi_y = Psi_y./ repmat(sqrt(sum(Psi_y.^2)) , [size(Psi_y ,1), 1]) ;  
				end
				
				D = [Psi_cxLR, Psi_xLR, zeros(N,K) ; ...
							Psi_cy, zeros(N,K), Psi_y];
						
				D = D./ repmat(sqrt(sum(D.^2)) , [size(D ,1), 1]) ; 
				Z = omp(D' * data, D' * D, s_c+s_x+s_y); 		
				Z_cfound = Z(ind1, :);
				Z_xfound = Z(ind2, :);
				Z_yfound = Z(ind3, :);
				
				
			otherwise
				disp('No matching method')				
		end
		% Recovery ends. Compute recovered testing dataset.
	
		X_rec = Psi_cx*Z_cfound + Psi_x*Z_xfound;
		Y_rec = Psi_cy*Z_cfound + Psi_y*Z_yfound;

		% measure time
		timeElapsed = toc;

		% --------------------------------------------------------

		% compute sparsity for each patch with fixed no-zero threshold
		S_vec = [];
		S_vec(1,:) = sort( sum(abs(Z_cfound)>1e-5 , 1), 'descend');
		S_vec(2,:) = sort( sum(abs(Z_xfound)>1e-5 , 1), 'descend' );
		S_vec(3,:) = sort( sum(abs(Z_yfound)>1e-5 , 1), 'descend' );

		mean_s = zeros(3,1);
		mean_s(1) = mean( S_vec(1,:));
		mean_s(2) = mean( S_vec(2,:));
		mean_s(3) = mean( S_vec(3,:));	

% 		% add smooth patches into
% 		X_rec2 = X_test_vecLR0;
% 		X_rec2(:, indCol) = X_rec;
% 		Y_rec2 = Y_test_vec0;
% 		Y_rec2(:, indCol) = Y_rec;
% 		
% 		X_rec = X_rec2;
% 		Y_rec = Y_rec2;
% 		Y_test_vec = Y_test_vec0;
		
		% compute recovery error for each patch.
		ErrPatch_X = zeros(1, T);
		ErrPatch_Y = zeros(1, T);
		ErrBlocksize = 2000;	% compute error in blocks to conserve memory
		for i = 1:ErrBlocksize: T
			blockids = i : min(i+ErrBlocksize-1, T);
			ErrPatch_X(blockids) = sum( (X_test_vec(:,blockids) - X_rec(:,blockids)).^2 );
			ErrPatch_Y(blockids) = sum( (Y_test_vec(:,blockids) - Y_rec(:,blockids)).^2 );
		end
		
% 		% After reconstructing the centered HR patches, we add the mean back to them. 
% 		X_rec_mean = X_rec + repmat(dc_XLR, size(X_rec,1), 1);
% 		X_rec_im = col2imstep(X_rec_mean, cropwidth, blocksize, stepsize);	
% 
% 		Y_rec_mean = Y_rec + repmat(dc_Y, size(Y_rec,1), 1);
% 		Y_rec_im = col2imstep(Y_rec_mean, cropwidth, blocksize, stepsize);	

		% average over the overlapping blocks
		X_rec_im = patches2video_fast(X_rec, 512, 512, 3, 4, 4);
%         X_rec_im = patches2video_fast(X_rec, 512, 512, 3, 2, 2);
		Y_rec_im = X_rec_im(:,:,1);
		
% 		% average over the overlapping blocks of the separated signals
% 		cnt = countcover(cropwidth,blocksize,stepsize);
% 		for i = 1:size(cnt,1)
% 			for j = 1:size(cnt,2)
% 				if cnt(i,j) == 0
% 					cnt(i,j) = 1;
% 				end
% 			end
% 		end
% 		X_rec_im = X_rec_im./cnt; 
% 		Y_rec_im = Y_rec_im./cnt;
		
% 		X_rec_im(X_rec_im > 1) = 1;
% 		X_rec_im(X_rec_im < 0) = 0;
% 		Y_rec_im(Y_rec_im > 1) = 1;
% 		Y_rec_im(Y_rec_im < 0) = 0;	

	% --------------------------------------------------------
	%% Recover With No Side Information via interpolation

		% LR image is obtained via bicubic interpolation,too.
		X_low = imresize(X_test, 1/upscale, 'bicubic');

		% generate interpolated image;
		interpolated = imresize(X_low, upscale, 'bicubic');     % bicubic, bilinear
% 		interpolated = interpolated(1: cropwidth(1), 1: cropwidth(2));

% 		% shave the border
% 		ImALL = cat(3, X_rec_im, Y_rec_im, interpolated, X_test, Y_test );
%         ImALL = shave(ImALL, [upscale,upscale]);
% 		
% 		X_rec_im = ImALL(:,:,1);
% 		Y_rec_im = ImALL(:,:,2);
% 		interpolated = ImALL(:,:,3);
% 		X_test = ImALL(:,:,4);
% 		Y_test = ImALL(:,:,5);			
		
		%%
		% comput PSNR, RMSE
		ImgX = X_rec_im;
		RefX = X_test;
		[PSNR_X, SNR_X] = psnr(ImgX,RefX); % or [PSNR_X, SNR_X] = psnr(ImgX,RefX,peakval); 
		MSE_X = sum(sum((ImgX-RefX).^2)) ./ numel(RefX);
		RMSE_X = mean(sqrt(MSE_X));
% 		Error_X = norm(RefX - ImgX,'fro')/ norm(RefX,'fro'); 


		ImgY = Y_rec_im;
		RefY = X_test(:,:,1);
		[PSNR_Y, SNR_Y] = psnr(ImgY,RefY)	;
		MSE_Y = sum(sum((ImgY-RefY).^2)) ./ numel(RefY);
		RMSE_Y = sqrt(MSE_Y);
% 		Error_Y = norm(RefY -ImgY,'fro')/ norm(RefY ,'fro');

		% 	show results
		disp('----------------------')
		info = sprintf('With Side Info (image):  \n');
		info = sprintf('%s PSNR_X = %.4f, RMSE_X = %.4f, SNR_X = %.4f ; \n', info, PSNR_X, RMSE_X, SNR_X);
		info = sprintf('%s PSNR_Y = %.4f, RMSE_Y = %.4f, SNR_Y = %.4f ; ', info, PSNR_Y, RMSE_Y, SNR_Y);
		disp(info);

		PSNRall(i_MeasNo).PSNR_X = PSNR_X;
		PSNRall(i_MeasNo).PSNR_Y = PSNR_Y;
		PSNRall(i_MeasNo).RMSE_X = RMSE_X;
		PSNRall(i_MeasNo).RMSE_Y = RMSE_Y;
		ImgRec(i_MeasNo).X = uint8(255.*X_rec_im);
		ImgRec(i_MeasNo).Y = uint8(255.*Y_rec_im);
		PSNRall(i_MeasNo).mean_s = mean_s;
	% --------------------------------------------------------

		% comput PSNR, RMSE
		ImgX = interpolated;
		RefX = X_test;

		[PSNR_X, SNR_X] = psnr(ImgX,RefX); 
		MSE_X = sum(sum((ImgX-RefX).^2)) ./ numel(RefX);
		RMSE_X = mean(sqrt(MSE_X));
% 		Error_X = norm(RefX - ImgX,'fro')/ norm(RefX,'fro'); 

		% 	show results
		info = sprintf('After interpolation (image):  \n ');
		info = sprintf('%s PSNR_X = %.4f, RMSE_X = %.4f, SNR_X = %.4f ; ', info, PSNR_X, RMSE_X, SNR_X);
		disp(info);

		PSNRall(i_MeasNo).PSNR_X_interpolated = PSNR_X;
		PSNRall(i_MeasNo).RMSE_X_interpolated = RMSE_X;
		ImgRec(i_MeasNo).X_interpolated = uint8(255.*interpolated);
		ImgRec(i_MeasNo).X_low = uint8(255.*X_low);
		
		
		%% compute SSIM
		mssim = []; ssim_map = {};
		interpolated = im2uint8(interpolated); 
		X_rec_im = im2uint8(X_rec_im); 
		X_test = im2uint8(X_test); 
		[mssim(1), ssim_map{1}] = ssim(interpolated, X_test) ;
		[mssim(2), ssim_map{2}] = ssim(X_rec_im, X_test) ;
	end

	outputCSR{img_index}.ImgRec = ImgRec;
	outputCSR{img_index}.PSNRall = PSNRall;
	outputCSR{img_index}.mssim = mssim;
	outputCSR{img_index}.ssim_map = ssim_map;
	outputCSR{img_index}.info = info;
	outputCSR{img_index}.mean_s = mean_s;
	outputCSR{img_index}.timeElapsed = timeElapsed;
	
	% store recovery results
	SIZE = ['_D',num2str(N),'x',num2str(K)];
	SC = ['_sc', num2str(s_c(1))];
	SX = ['sx', num2str(s_x(1))];
	SY = ['sy', num2str(s_y(1))];
		
	TestSize = ['_T', num2str(T)];
	ImgNo = [ '_No', num2str(img_index)];
	Meas = ['_Meas', num2str(MeasNo)] ;
	Scale = ['_Scale', num2str(upscale)];
	Step = ['_Step', num2str(stepsize(1))];
	current_date = date;
	DATE = ['_Date',current_date];

	FILENAME = ['ImgCSR', SIZE, SC, SX, SY, TestSize, Scale,Step, '_', Method_RecoverStage,DATE];

	save([FILENAME,'.mat'], ...	
		'paramsCSR', 'paramsCDL', ...
		'outputCSR')
end



%% summarize the results and compute the mean
for i = 1 : numel(outputCSR)
	
	PSNR_array(i) = outputCSR{i}.PSNRall.PSNR_X ;
	PSNR_interpolated_array(i) = outputCSR{i}.PSNRall.PSNR_X_interpolated ;
	PSNR_Y_array(i) = outputCSR{i}.PSNRall.PSNR_Y ;

	RMSE_array(i) = outputCSR{i}.PSNRall.RMSE_X ;
	RMSE_interpolated_array(i) = outputCSR{i}.PSNRall.RMSE_X_interpolated ;
	RMSE_Y_array(i) = outputCSR{i}.PSNRall.RMSE_Y ;
	
	MSSIM_array(i) = outputCSR{i}.mssim(2);
	MSSIM_interpolated_array(i) = outputCSR{i}.mssim(1);

end

outputCSRsum.PSNR_array = PSNR_array ;
outputCSRsum.PSNR_Y_array = PSNR_Y_array ;
outputCSRsum.PSNR_interpolated_array = PSNR_interpolated_array ;
outputCSRsum.RMSE_array = RMSE_array ;
outputCSRsum.RMSE_Y_array = RMSE_Y_array ;
outputCSRsum.RMSE_interpolated_array = RMSE_interpolated_array ;
outputCSRsum.MSSIM_array = MSSIM_array ;
outputCSRsum.MSSIM_interpolated_array = MSSIM_interpolated_array ;

outputCSRsum.PSNR_mean = mean(PSNR_array) ;
outputCSRsum.PSNR_Y_mean = mean(PSNR_Y_array) ;
outputCSRsum.PSNR_interpolated_mean = mean(PSNR_interpolated_array) ;
outputCSRsum.RMSE_mean = mean(RMSE_array) ;
outputCSRsum.RMSE_Y_mean = mean(RMSE_Y_array) ;
outputCSRsum.RMSE_interpolated_mean = mean(RMSE_interpolated_array) ;
outputCSRsum.MSSIM_mean = mean(MSSIM_array) ;
outputCSRsum.MSSIM_interpolated_mean = mean(MSSIM_interpolated_array) ;


save([FILENAME,'.mat'], ...	
	'paramsCSR', 'paramsCDL', ...
	'outputCSR', 'outputCSRsum')

disp('****************************************************')

%% show results
results = [ [outputCSRsum.MSSIM_interpolated_array'; outputCSRsum.MSSIM_interpolated_mean], ...
	[outputCSRsum.PSNR_interpolated_array'; outputCSRsum.PSNR_interpolated_mean], ...
	[outputCSRsum.MSSIM_array'; outputCSRsum.MSSIM_mean], ...
	[outputCSRsum.PSNR_array'; outputCSRsum.PSNR_mean] ];

disp('done!')



