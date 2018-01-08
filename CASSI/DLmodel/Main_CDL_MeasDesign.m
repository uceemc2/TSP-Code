% =========================================================================
% 
% 31/08/2017 Super-resolving RGB images with NIR images as guidance information.
% The measurement process is not blurring + downsampling. Instead, each measurement is a predifined linear combination of an image patch. The linear combination setting is included in the measurement matrix design from Mengyang et al.'s GMM work.
% 
% This is the faster version of coupled dictionary learning. Please use
% this version to train larger dictionaries instead of using Main_CDL_CoupledKSVD.m. 
% 
% Multimodal Image Super-resolution via Joint Sparse Representations 
% induced by Coupled Dictionaries 

% 
% Data Model:  
% 
%      XL = Psi_cxLR*Z_c + Psi_xLR*Z_x
%      X = Psi_cx*Z_c + Psi_x*Z_x
%      Y = Psi_cy*Z_c + Psi_y*Z_y
%
% where:
% XL: N x T; Low-resolution (LR) image patches.
% X: N x T; High-resolution (HR) image patches.
% Y: N x T; High-resolution (HR) guidance image patches.
% Psi_cx, Psi, Psi_cy, and Psi_y (in NxK) are the coupled dictionaries, 
% Z_c, Z_x, and Z_y (in KxT) are the sparse codes; 
% 
% 
% Goal: The goal is to learn these dictionaries;
% 
% Approach: Coupled Dictionary Learning (CDL) 
% Step 1: learn Psi_cxLR, Psi_cy, Psi_xLR, Psi_y via solving
%		min_{Psi_cxLR, Psi_cy, Psi_xLR, Psi_y, Z_c, Z_x, Z_y} 
%				|| [XL ; Y] - [Psi_cxLR; Psi_cy] * Z_c - [Psi_xLR, 0; 0, Psi_y] * [Z_x; Z_y] ||_F^2;
%		s.t. ||z_ci||_0 <= s_c; ||z_xi||_0 <= s_x; ||z_yi||_0 <= s_y; \forall i
% We adapted K-SVD to alternatively train common dicts and unique dicts. 
% 
% Step2: learn Psi_cx, Psi_x based on learned LR dicts and HR guidance dicts and the sparse codes
% 		Gam_L = [Z_c; Z_x];
% 		[Psi_cx, Psi_x] = (X * Gam_L') * inv(full(Gam_L * Gam_L')); 
% 
% References
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
clear; close all
addpath('../ksvd');
addpath('../ksvd/ksvdbox');
addpath('../ksvd/ksvdbox/private_ccode');
addpath('../ksvd/ompbox');
% addpath('../TestImages_Multispectral');
% addpath('../TrainImages_Multispectral');
addpath('./utils')
addpath('./utils2')

fprintf('Coupled dictionary learning from multispectral and RGB images. \n');
for KK = 3
% parameters
	switch KK
		case 1
			s_c = 2;
			s_x = 1;
			s_y = 1;		
		case 2
			s_c = 3;
			s_x = 1;
			s_y = 1;	
		case 3
			s_c = 4;
			s_x = 2;
			s_y = 2;			
	end

% 	% Random seed number
% 	seed = RandStream('mcg16807','Seed',31415);
% 	RandStream.setGlobalStream(seed);
% % 	seed = RandStream.getGlobalStream;


	% Dictionary learning parameters
	Modality = 'MultispectRGB';
	K = 512; % The number of atoms.
	NXL = 64; % The length of one atom for LR dicts.
	NY = 32;  % The length of one atom for side info dicts
	NXH = 192; % The length of one atom for HR dicts
	
	Main_Iternum = 10 ;  % cycle for training of common dicts + unique dicts;	
	Sub_Iternum = 10 ; % iteration for training of either common dicts or unique dicts. 
	MAX_ITER = Main_Iternum*Sub_Iternum; % Total iteration number.
	MONTE_CARLO = 1;
	trainnum = 5000; % 50000; % {500, 1000,5000,10000}
	blocksize = [sqrt(NXL),sqrt(NXL)]; % {[8,8],[8,8],[8,8]}	
	variance_Thresh = 0.02; % Discard those patch pairs with too small variance.
	upscale = 4;
	
	% fast test
% 	Main_Iternum = 2 ;  % cycle for training of common dicts + unique dicts;	
% 	Sub_Iternum = 5 ; % iteration for training of either common dicts or unique dicts. 
% 	MAX_ITER = Main_Iternum*Sub_Iternum; 
% 	MONTE_CARLO = 1;
% 	trainnum = 5000; 
% 	K = 64; 
	
% column and row positions of four small dictionaries in the whole dictionary. 
% 	the whole dictionary = 
% 	[Psi_cx,   Psi_x,            zeros(N,K) ; ...
% 	Psi_cy,    zeros(N,K),    Psi_y];
	ind1 = 1: K;
	ind2 = (K+1) : 2*K;
	ind3 = (2*K +1) : 3*K;

	ind1_row = 1: NXL;
	ind2_row = (NXL + 1) : 2*NXL;

	% Select available training methods:
	Method_cell = {'ksvd_JointTrain'};
	Method_Stage1 = Method_cell{1}; 
	Method_Stage2 = Method_cell{1}; 
	
% --------------------------------------------------------
% 	% Load training image and construct training dataset.
% 	directory = '../TrainImages_Multispectral'; 
% 	patternX = '*.png';
% 	patternY = '*.bmp';
% 
% 	XpathCell = glob(directory, patternX );
% 	Xcell = load_images( XpathCell );
% 
% 	YpathCell = glob(directory, patternY );
% 	Ycell = load_images(YpathCell );
% 
% 	if length(Xcell) ~= length(Ycell)	
% 		error('Error: The number of X images is not equal to the number of Y images!');
% 	end

	
	paramsCDL.K = K ;
	paramsCDL.NXL = NXL ;
	paramsCDL.NXH = NXH ;
	paramsCDL.NY = NY ;
	paramsCDL.S = [s_c, s_x, s_y] ;
	paramsCDL.S_total = s_c + s_x + s_y ;
	paramsCDL.MONTE_CARLO = MONTE_CARLO ;
	paramsCDL.MAX_ITER = MAX_ITER ;
	paramsCDL.Main_Iternum = Main_Iternum;
	paramsCDL.Sub_Iternum = Sub_Iternum;
	paramsCDL.trainnum = trainnum ;
	paramsCDL.blocksize = blocksize ;
	paramsCDL.variance_Thresh = variance_Thresh ;
% 	paramsCDL.XpathCell = XpathCell;
% 	paramsCDL.YpathCell = YpathCell;
	paramsCDL.upscale = upscale;
% 	paramsCDL.Xcell = Xcell;
% 	paramsCDL.Ycell = Ycell;
	paramsCDL.filter_flag = 0 ; % 1: perform high-pass filtering on the LR images. 0: not.

	% Load training image and construct training dataset.
	% Use RGB images with .bmp as the signal of interest and another wavelength as the SI. 
% 	[Xh, Xl, Yh] = Sample_PreProcess( paramsCDL);
% 
% 	X = Xh;
% 	XL = Xl; % may need to shrink the LR features
% 	Y = Yh;
% 	clear Xh Xl Yh
% 
% 	T = size(X,2); % training data size
% 
% 	paramsCDL.T = T ;
% 	% remove data field to reduce the storage ccost.
% 	paramsCDL = rmfield(paramsCDL,'Xcell'); 
% 	paramsCDL = rmfield(paramsCDL,'Ycell');
	

	X = []; XL = []; Y = [];
	TrainImgNo = [2, 3, 4, 6, 7, 8];
	
	% (1) construct training data
	for i = 1: length(TrainImgNo)
		ImgName = ['scene', num2str(TrainImgNo(i))];
		[P1, xh, xl, P2, yh, yl] = DLmodeltraining(ImgName);
        yl=0.5*yl;
		X = [X, xh];
		XL = [XL, xl];
		Y = [Y, yl];
	end

% 	% (2) load training dataset
% 	load('TrainData.mat')
	
% 	% remove the DC components
% 	dc_Y = mean(Y);
% 	dc_XL = mean(XL);
% 
% 	Y = Y - repmat(dc_Y, size(Y, 1), 1);
% 	XL = XL - repmat(dc_XL, size(XL, 1), 1);		
% 	% Xh should remove mean referring to XL;
% 	X = X - repmat(dc_XL, size(X, 1), 1);
% 	
% 	DataNorm(1, : ) = sum(XL.^2);
% 	DataNorm(2, : ) = sum(X.^2);
% 	DataNorm(3, : ) = sum(Y.^2);
% 	[~, indexNorm] = sort( DataNorm(2, : ) );
% 	DataNorm = DataNorm(:, indexNorm );
% 	figure; plot(DataNorm')
	
	T = size(X,2); % training data size
	paramsCDL.T = T ;
	
	
% --------------------------------------------------------	
%% Dictionary learning			

% 	% open parallel pool
% 	mypool = gcp;
% 	if isempty(mypool)
% 		mypool = parpool() % matlabpool open
% 	end

	outputCDL = [];
	for mc = 1: MONTE_CARLO

		info = sprintf('CDL mc = %d / %d begin \n', mc, MONTE_CARLO);
		info = sprintf('%s Param: s_c = %d, s_x = %d, s_y = %d,', info, s_c, s_x, s_y);
		info = sprintf('%s training data size = %d, dict size = %dx%d,', info, T, NXL, K);
		info = sprintf('%s Method: %s', info, Method_Stage1);
		disp(info);
		info_start = info;
	

% 	% Initialize the dictionaries and sparse representation
% 	Initialization method 1: using random matrix
		Psi_cxLR = randn(NXL,K) ;
		Psi_xLR = randn(NXL,K) ;
		Psi_cy = randn(NY,K) ;
		Psi_y = randn(NY,K) ;
		
		Psi_cxLR = Psi_cxLR./ repmat(sqrt(sum(Psi_cxLR.^2)) , [size(Psi_cxLR ,1), 1]) ; 
		Psi_xLR = Psi_xLR./ repmat(sqrt(sum(Psi_xLR.^2)) , [size(Psi_xLR ,1), 1]) ; 
		Psi_cy = Psi_cy./ repmat(sqrt(sum(Psi_cy.^2)) , [size(Psi_cy ,1), 1]) ; 
		Psi_y = Psi_y./ repmat(sqrt(sum(Psi_y.^2)) , [size(Psi_y ,1), 1]) ; 
		
		Z_c = zeros(K, T);
		Z_x = zeros(K, T);
		Z_y = zeros(K, T);		


% % 	Initialization method 2: using DCT matrix
% 		bb = 8;
% 		Pn=ceil(sqrt(K));
% 		DCT=zeros(bb,Pn);
% 		for k=0:1:Pn-1,
% 			V=cos([0:1:bb-1]'*k*pi/Pn);
% 			if k>0, V=V-mean(V); end;
% 			DCT(:,k+1)=V/norm(V);
% 		end;
% 		DCT=kron(DCT,DCT);
% 
% 		Psi_cxLR = DCT ;
% 		Psi_xLR = DCT ;
% 		Psi_cy = DCT ;
% 		Psi_y = DCT ;
% 		
% 		Psi_cxLR = Psi_cxLR./ repmat(sqrt(sum(Psi_cxLR.^2)) , [size(Psi_cxLR ,1), 1]) ; 
% 		Psi_xLR = Psi_xLR./ repmat(sqrt(sum(Psi_xLR.^2)) , [size(Psi_xLR ,1), 1]) ; 
% 		Psi_cy = Psi_cy./ repmat(sqrt(sum(Psi_cy.^2)) , [size(Psi_cy ,1), 1]) ; 
% 		Psi_y = Psi_y./ repmat(sqrt(sum(Psi_y.^2)) , [size(Psi_y ,1), 1]) ; 
% 
% 		Z_c = 0.5*(Psi_cxLR'*XL + Psi_cy'*Y);
% 		Z_x = Psi_xLR'*XL;
% 		Z_y = Psi_y'*Y;
	%------------------------------------------------------------------------------
	
		% set training parameters	
		params.initdict = [Psi_cxLR; Psi_cy];
		params.data = [XL; Y];
		params.codemode = 'sparsity';
		params.Tdata = s_c;
		params.iternum = MAX_ITER;
		params.memusage = 'high';	%	'normal';
		params.RowIndex = [NXL, NY];

		disp('Train common dicts Psi_cxLR and Psi_cy;')
		
		tic; % measure time
		[Dict,Z_c, Err_c, gerr, MoreOutput] = ksvd_JointTrain(params,'irt');
		timeElapsed = toc; % measure elapsed time
		
		disp(['Elapsed time: ', num2str(timeElapsed)]);
		
		
		info_long = MoreOutput.info_all;
		Conv = MoreOutput.RMSE ;

		% comput PSNR, RMSE				
		Psi_cxLR = Dict( 1:NXL , : );
		Psi_cy = Dict( (1:NY)+NXL, :);
		
		XL_rec = Psi_cxLR*Z_c ;
		Y_rec = Psi_cy*Z_c ;

		% psnr = 10*log10(peakval^2/MSE); peakval=1 for double, = 255 for uint8
		% snr = 10*log10( norm(REF,'fro')/ norm(REF - Img,'fro') ).
		ImgX = XL_rec;
		RefX = XL;

		[PSNR_cXL, SNR_cXL] = psnr(ImgX,RefX); % or [PSNR_cX, SNR_cX] = psnr(ImgX,RefX,peakval); 
		MSE_cXL = sum(sum((ImgX-RefX).^2)) ./ numel(RefX);
		RMSE_cXL = sqrt(MSE_cXL);
		Error_cXL = norm(RefX - ImgX,'fro')/ norm(RefX,'fro'); % SNR_cX = 20*log10(1/Error_cX)

		ImgY = Y_rec;
		RefY = Y;
		[PSNR_cY, SNR_cY] = psnr(ImgY,RefY)	;
		MSE_cY = sum(sum((ImgY-RefY).^2)) ./ numel(RefY);
		RMSE_cY = sqrt(MSE_cY);
		Error_cY = norm(RefY -ImgY,'fro')/ norm(RefY ,'fro'); % SNR_cY = 20*log10(1/Error_cY)

		info = sprintf(' Training Results for coupled dictionaries: \n'); 
		info = sprintf('%s PSNR_cXL = %.4f, RMSE_cXL = %.4f, SNR_cXL = %.4f, Error_cXL = %.4f ; \n', info, PSNR_cXL, RMSE_cXL, SNR_cXL, Error_cXL);
% 		info = sprintf('%s PSNR_cX = %.4f, RMSE_cX = %.4f, SNR_cX = %.4f, Error_cX = %.4f ; \n', info, PSNR_cX, RMSE_cX, SNR_cX, Error_cX);
		info = sprintf('%s PSNR_cY = %.4f, RMSE_cY = %.4f, SNR_cY = %.4f, Error_cY = %.4f ; \n', info, PSNR_cY, RMSE_cY, SNR_cY, Error_cY);
		disp(info);
		info_ComD = info;
	%------------------------------------------------------------------------------
	
	% train innovation dictionaries	Psi_xLR
		clear params
		params.initdict = Psi_xLR;
		params.data = XL - Psi_cxLR *Z_c;
		params.iternum = MAX_ITER;
		params.codemode = 'sparsity';
		params.Tdata = s_x;
		params.memusage = 'high';	%	'normal'; % 'high'
% 		params.testdata = Xcvx;

		disp('Train innovation dict Psi_xLR;')
		[Psi_xLR, Z_x, Err_x, gerr] = ksvd(params,'irt');
		
		XL_rec2 = Psi_xLR*Z_x;
		
		% psnr = 10*log10(peakval^2/MSE); peakval=1 for double, = 255 for uint8
		% snr = 10*log10( norm(REF,'fro')/ norm(REF - Img,'fro') ).
		ImgX = XL_rec2;
		RefX = XL - Psi_cxLR*Z_c;

		[PSNR_X, SNR_X] = psnr(ImgX,RefX); % or [PSNR_X, SNR_X] = psnr(ImgX,RefX,peakval); 
		MSE_X = sum(sum((ImgX-RefX).^2)) ./ numel(RefX);
		RMSE_X = sqrt(MSE_X);
		Error_X = norm(RefX - ImgX,'fro')/ norm(RefX,'fro'); % SNR_X = 20*log10(1/Error_X)

		info = sprintf('Training Results: \n'); 
% 		info = sprintf('%s PSNR_XL = %.4f, RMSE_XL = %.4f, SNR_XL = %.4f, Error_XL = %.4f ; \n', info, PSNR_XL, RMSE_XL, SNR_XL, Error_XL);
		info = sprintf('%s PSNR_X = %.4f, RMSE_X = %.4f, SNR_X = %.4f, Error_X = %.4f ; \n', info, PSNR_X, RMSE_X, SNR_X, Error_X);
		disp(info);

		 %------------------------------------------------------------------------------
		% train innovation dictionaries	Psi_y
		clear params
		params.initdict = Psi_y;
		params.data = Y - Psi_cy*Z_c;
		params.iternum = MAX_ITER;
		params.codemode = 'sparsity';
		params.Tdata = s_y;	
		params.memusage = 'high';	%	'normal'; % 'high'
% 		params.testdata = Ycvy;

		disp('Train innovation dict Psi_y;')
		[Psi_y, Z_y, Err_y, gerr] = ksvd(params,'irt');

		Y_rec2 = Psi_y*Z_y;

% 			SRC = nnz(abs(Gam)>1e-5)/size(Gam,2); % sparse representation capacity;
% 			PSNR_pre = 20*log10(1./ RMSE_y); % predefined PSNR
		ImgY = Y_rec2;
		RefY = Y - Psi_cy*Z_c;

		[PSNR_Y, SNR_Y] = psnr(ImgY,RefY)	;
		MSE_Y = sum(sum((ImgY-RefY).^2)) ./ numel(RefY);
		RMSE_Y = sqrt(MSE_Y);
		Error_Y = norm(RefY -ImgY,'fro')/ norm(RefY ,'fro'); % SNR_Y = 20*log10(1/Error_Y)

		info = sprintf('Training Results: \n'); 
		info = sprintf('%s PSNR_Y = %.4f, RMSE_Y = %.4f, SNR_Y = %.4f, Error_Y = %.4f ; \n', info, PSNR_Y, RMSE_Y, SNR_Y, Error_Y);
		disp(info);

	%------------------------------------------------------------------------------
		% sout out the entire results
		Gamma = [Z_c; Z_x; Z_y] ;

		% compute sparsity for each patch with fixed no-zero threshold
		S_vec = [];
		S_vec(1,:) = sort( sum(abs(Z_c)>1e-5 , 1), 'descend');
		S_vec(2,:) = sort( sum(abs(Z_x)>1e-5 , 1), 'descend' );
		S_vec(3,:) = sort( sum(abs(Z_y)>1e-5 , 1), 'descend' );

		mean_s = zeros(3,1);
		mean_s(1) = mean( S_vec(1,:));
		mean_s(2) = mean( S_vec(2,:));
		mean_s(3) = mean( S_vec(3,:));	
		MeanS = mean_s;

		% compute HR dicts Psi_cx,Psi_x based on LR dicts and HR dicts and the sparse codes
		Gam_L = [Z_c; Z_x];
		DictH = (X * Gam_L') * inv(full(Gam_L * Gam_L')); 
		Psi_cx = DictH(:, ind1);
		Psi_x = DictH(:, ind2);
		
		XL_rec = Psi_cxLR * Z_c + Psi_xLR * Z_x;
		X_rec = Psi_cx * Z_c + Psi_x * Z_x;
		Y_rec = Psi_cy * Z_c + Psi_y * Z_y;

		
		% compute the average usage rate and average contribution for each atoms.
		% We need half the common coefficients Z_c, because the norm of common dicts 
		% is shrinked during dictionary learning, accordingly the coefficients are enlarged. 
		AtomUsage.Xcom = full(sum(abs(Z_c./2)>1e-7 , 2)/ T);
		AtomUsage.Xuniq = full(sum(abs(Z_x)>1e-7 , 2)/ T);
		AtomUsage.Ycom = AtomUsage.Xcom;
		AtomUsage.Yuniq = full(sum(abs(Z_y)>1e-7 , 2)/ T);
		
		AtomUsage.Xcom_Contrib = full(sum(abs(Z_c./2), 2));
		AtomUsage.Xuniq_Contrib = full( sum(abs(Z_x), 2));
		AtomUsage.Ycom_Contrib = AtomUsage.Xcom_Contrib;
		AtomUsage.Yuniq_Contrib = full(sum(abs(Z_y), 2));
		
% % 	SRC = nnz(abs(Gam)>1e-5)/size(Gam,2); % sparse representation capacity;		
% 		% psnr = 10*log10(peakval^2/MSE); peakval=1 for double, = 255 for uint8
% 		% snr = 10*log10( norm(REF,'fro')/ norm(REF - Img,'fro') ).		
		ImgX = XL_rec;
		RefX = XL;

		[PSNR_XL, SNR_XL] = psnr(ImgX,RefX); % or [PSNR_X, SNR_X] = psnr(ImgX,RefX,peakval); 
		MSE_XL = sum(sum((ImgX-RefX).^2)) ./ numel(RefX);
		RMSE_XL = sqrt(MSE_XL);
		Error_XL = norm(RefX - ImgX,'fro')/ norm(RefX,'fro'); % SNR_X = 20*log10(1/Error_X)

		
		ImgX = X_rec;
		RefX = X;

		[PSNR_XH, SNR_XH] = psnr(ImgX,RefX); % or [PSNR_X, SNR_X] = psnr(ImgX,RefX,peakval); 
		MSE_XH = sum(sum((ImgX-RefX).^2)) ./ numel(RefX);
		RMSE_XH = sqrt(MSE_XH);
		Error_XH = norm(RefX - ImgX,'fro')/ norm(RefX,'fro'); % SNR_XH = 20*log10(1/Error_X)


		ImgY = Y_rec;
		RefY = Y;
		[PSNR_Y, SNR_Y] = psnr(ImgY,RefY)	;
		MSE_Y = sum(sum((ImgY-RefY).^2)) ./ numel(RefY);
		RMSE_Y = sqrt(MSE_Y);
		Error_Y = norm(RefY -ImgY,'fro')/ norm(RefY ,'fro'); % SNR_Y = 20*log10(1/Error_Y)

		info = sprintf('Training Results for MONTE_CARLO = %d / %d: \n', mc, MONTE_CARLO); 
		info = sprintf('%s PSNR_XL = %.4f, RMSE_XL = %.4f, SNR_XL = %.4f, Error_XL = %.4f ; \n', info, PSNR_XL, RMSE_XL, SNR_XL, Error_XL);
		info = sprintf('%s PSNR_XH = %.4f, RMSE_XH = %.4f, SNR_XH = %.4f, Error_XH = %.4f ; \n', info, PSNR_XH, RMSE_XH, SNR_XH, Error_XH);
		info = sprintf('%s PSNR_Y = %.4f, RMSE_Y = %.4f, SNR_Y = %.4f, Error_Y = %.4f ; \n', info, PSNR_Y, RMSE_Y, SNR_Y, Error_Y);
		info = sprintf(['%s','Elapsed time: ', num2str(timeElapsed)], info);
		disp(info); 
		disp('=====================================================');		
		info_end = info;
		info_short = sprintf([info_start, '\n', info_ComD, info_end]);
	
		SNR.XL = SNR_XL;
		SNR.XH = SNR_XH;
		SNR.Y = SNR_Y ;	
		
		RMSE.XY = Conv.XY(end);
		RMSE.X = Conv.X(end);
		RMSE.Y = Conv.Y(end);% 	RMSE.Y = RMSE_Y ;
		RMSE.XL = RMSE_XL ;
		RMSE.XH = RMSE_XH ;

		
		PSNR.XY = 20*log10(1/Conv.XY(end)) ;
		PSNR.XL = PSNR_XL ;
		PSNR.XH = PSNR_XH ;
		PSNR.Y = PSNR_Y ;		
		
		% we can also compute the using the following formulations.
% 		peakval = 1;
% 		PSNR.XY = 20*log10(peakval^2/Conv.XY(end)) ;
% 		PSNR.X = 20*log10(peakval^2/Conv.X(end)) ;
% 		PSNR.Y = 20*log10(peakval^2/Conv.Y(end)) ;	

%% Test
	% use OMP package, but the sparse threshold can not be larger than measurements number
		data = [XL; Y] ; 
		D = [Psi_cxLR; Psi_cy];
		D = D./ repmat(sqrt(sum(D.^2)) , [size(D ,1), 1]) ; 
		s_c =10;  s_x = 2; s_y = 2;
		Z_cfound = omp(D' * data, D' * D, s_c); % fast		
% 				PP = inv(D' * D + lambda*eye(size(D,2))) * D' ; Z_cfound = PP * data;	% Use ridge regresson instead of OMP.

		X_lowR = XL - Psi_cxLR*Z_cfound;
		Y_highR = Y - Psi_cy*Z_cfound;
		Z_xfound = omp(Psi_xLR' * X_lowR, Psi_xLR' *Psi_xLR, s_x);  % fast omp
% 		Z_xfound = omp(Psi_xLR, X_lowR, Psi_xLR' *Psi_xLR, s_x); 	% slow omp

		if sum( sum(Psi_y.^2) == 0) 
			colZero = find(sum(Psi_y.^2) == 0); 
			Psi_y(:, colZero) = sqrt(1/size(Psi_y,1)); 
			Psi_y = Psi_y./ repmat(sqrt(sum(Psi_y.^2)) , [size(Psi_y ,1), 1]) ;  
		end
		Z_yfound = omp(Psi_y' * Y_highR, Psi_y'*Psi_y, s_y); 

		XL_rec3 = Psi_cxLR * Z_cfound + Psi_xLR * Z_xfound;
		X_rec3 = Psi_cx * Z_cfound + Psi_x * Z_xfound;
		Y_rec3 = Psi_cy * Z_cfound + Psi_y * Z_yfound;

		ImgX = XL_rec3; 		RefX = XL;
		[PSNR_XL, SNR_XL] = psnr(ImgX,RefX); % or [PSNR_X, SNR_X] = psnr(ImgX,RefX,peakval); 
		MSE_XL = sum(sum((ImgX-RefX).^2)) ./ numel(RefX);
		RMSE_XL = sqrt(MSE_XL);
		Error_XL = norm(RefX - ImgX,'fro')/ norm(RefX,'fro'); % SNR_X = 20*log10(1/Error_X)

		ImgX = X_rec3; 		RefX = X;
		[PSNR_XH, SNR_XH] = psnr(ImgX,RefX); % or [PSNR_X, SNR_X] = psnr(ImgX,RefX,peakval); 
		MSE_XH = sum(sum((ImgX-RefX).^2)) ./ numel(RefX);
		RMSE_XH = sqrt(MSE_XH);
		Error_XH = norm(RefX - ImgX,'fro')/ norm(RefX,'fro'); % SNR_XH = 20*log10(1/Error_X)

		ImgY = Y_rec3;		RefY = Y;
		[PSNR_Y, SNR_Y] = psnr(ImgY,RefY)	;
		MSE_Y = sum(sum((ImgY-RefY).^2)) ./ numel(RefY);
		RMSE_Y = sqrt(MSE_Y);
		Error_Y = norm(RefY -ImgY,'fro')/ norm(RefY ,'fro'); % SNR_Y = 20*log10(1/Error_Y)

		info_test = sprintf('Testing Results for MONTE_CARLO = %d / %d: \n', mc, MONTE_CARLO); 
		info_test = sprintf('%s PSNR_XL = %.4f, RMSE_XL = %.4f, SNR_XL = %.4f, Error_XL = %.4f ; \n', info_test, PSNR_XL, RMSE_XL, SNR_XL, Error_XL);
		info_test = sprintf('%s PSNR_XH = %.4f, RMSE_XH = %.4f, SNR_XH = %.4f, Error_XH = %.4f ; \n', info_test, PSNR_XH, RMSE_XH, SNR_XH, Error_XH);
		info_test = sprintf('%s PSNR_Y = %.4f, RMSE_Y = %.4f, SNR_Y = %.4f, Error_Y = %.4f ; \n', info_test, PSNR_Y, RMSE_Y, SNR_Y, Error_Y);
		disp(info_test); 
		disp('=====================================================');		
		info_short = sprintf([info_start, '\n', info_end, '\n', info_test]);
		
	%% SAVE the dictionaries

		SIZE = ['_D',num2str(size(X,1)),'x',num2str(K)];
		SC = ['_sc', num2str(s_c(1))];
		SX = ['sx', num2str(s_x(1))];
		SY = ['sy', num2str(s_y(1))];
		MaxIter = ['_Iter', num2str(MAX_ITER)];
		TrainSize = ['_T', num2str(T)];
		Scale = ['_Scale', num2str(upscale)];
		MonteCarlo = ['_MC',num2str(MONTE_CARLO)];

		current_date = date;
		DATE = ['_Date',current_date];

		FILENAME = ['CDL_CKSVD', SIZE, SC, SX, SY, MaxIter,TrainSize, Scale, DATE];

% 		outputCDL(mc).Dict = Dict;
% 		outputCDL(mc).Gamma = Gamma;
		outputCDL(mc).Conv = Conv;
		outputCDL(mc).MeanS = MeanS;
		outputCDL(mc).AtomUsage = AtomUsage;
		
		outputCDL(mc).Psi_cxLR = Psi_cxLR;
		outputCDL(mc).Psi_xLR = Psi_xLR;
		outputCDL(mc).Psi_cx = Psi_cx;
		outputCDL(mc).Psi_x = Psi_x;
		outputCDL(mc).Psi_cy = Psi_cy;
		outputCDL(mc).Psi_y = Psi_y;

		outputCDL(mc).RMSE = RMSE;
		outputCDL(mc).SNR = SNR;
		outputCDL(mc).PSNR = PSNR;
		outputCDL(mc).info_long = info_long;
		outputCDL(mc).timeElapsed = timeElapsed;
		outputCDL(mc).info_short = info_short;
		outputCDL(mc).FILENAME = FILENAME;	

		save([FILENAME,'.mat'],'outputCDL', 'paramsCDL');

    end
        
end

%%
% compute the mean of ratio , distance, time.
MC = length(outputCDL) ;
convXY_mean = [];
convX_mean = [];
convY_mean = [];
time_mean = 0;
for mc = 1: MC
	time_mean = time_mean + outputCDL(mc).timeElapsed;
	convXY_mean = [convXY_mean, outputCDL(mc).Conv.XY] ;
	convX_mean = [convX_mean, outputCDL(mc).Conv.X] ;
	convY_mean = [convY_mean, outputCDL(mc).Conv.Y] ;
end
time_mean = time_mean./MC ;
convXY_mean = mean(convXY_mean, 2)  ;
convX_mean = mean(convX_mean, 2)  ;
convY_mean = mean(convY_mean, 2)  ;

%% draw figures
addpath('../export_fig')
plot_enable = 1;
FigFontName = 'Times New Roman';
FigFontSize = 10;
SaveFig = 0 ; % save figure or not
FigFormatCell = { '.eps', '.fig'};


if plot_enable == 1

% show training error convergence
	Hcf = figure ;
	Hca = gca ;
	iter = 1: length(convXY_mean) ;
	plot(iter, convXY_mean, 'k-','LineWidth',2);hold on
	plot(iter, convX_mean, 'r--','LineWidth',2);hold on
	plot(iter, convY_mean, 'b--','LineWidth',2);hold on

	xlabel('iteration'); ylabel('RMSE Convergence');
	legend1=legend('mean', 'Modality A', 'Modality B', 'Location','best');
	grid on ; 	grid minor
	pause(0.2)

	set(gcf, 'Color', 'w'); % make the background to be white.
	FigPosition = [100 100 320 200];
	FigName = ['RMSE_Conv'];
	SetFigure_MultAxes(Hcf, Hca, FigPosition, FigFontName, FigFontSize, SaveFig, FigName, FigFormatCell) ;

	
	% show trained dictionaries
	NXL = paramsCDL.NXL;
	K = paramsCDL.K;
	blocksize = paramsCDL.blocksize;

	ind1 = 1: K;
	ind2 = (K+1) : 2*K;
	ind3 = (2*K +1) : 3*K;

	ind1_row = 1: NXL;
	ind2_row = (NXL + 1) : 2*NXL;

	mc = 1;
	Psi_cxLR = outputCDL(mc).Psi_cxLR ;
	Psi_xLR = outputCDL(mc).Psi_xLR ;
	Psi_cx = outputCDL(mc).Psi_cx ;
	Psi_x = outputCDL(mc).Psi_x ;
	Psi_cy = outputCDL(mc).Psi_cy ;
	Psi_y = outputCDL(mc).Psi_y ;
	upscale = paramsCDL.upscale;
	
	[~, index] = sort( sum(Psi_cxLR.^2) );
	Psi_cxLR = Psi_cxLR(:,index);
	Psi_cx = Psi_cx(:,index);
	Psi_cy = Psi_cy(:,index);
	
	[~, index] = sort( sum(Psi_x.^2) );
	Psi_xLR = Psi_xLR(:,index);
	Psi_x = Psi_x(:,index);
	
	[~, index] = sort( sum(Psi_y.^2) );
	Psi_y = Psi_y(:,index);	
	
	 AtomNorm(:,1) = sum(Psi_cxLR.^2);
	 AtomNorm(:,2) = sum(Psi_xLR.^2);
	 AtomNorm(:,3) = sum(Psi_cx.^2);
	 AtomNorm(:,4) = sum(Psi_x.^2);
	 AtomNorm(:,5) = sum(Psi_cy.^2);
	 AtomNorm(:,6) = sum(Psi_y.^2);
	 
	figure;
	plot(AtomNorm)
	
		
	dictimg1 = SMALL_showdict(Psi_cxLR,blocksize,round(sqrt( K )), round(sqrt( K )) ,'lines','highcontrast');  
	dictimg2 = SMALL_showdict(Psi_xLR,blocksize, round(sqrt( K )), round(sqrt( K )) ,'lines','highcontrast');  
	dictimg3 = SMALL_showdict(Psi_cx,[14,14],	round(sqrt( K )), round(sqrt( K )) ,'lines','highcontrast');  
	dictimg4 = SMALL_showdict(Psi_x,[14,14], round(sqrt( K )), round(sqrt( K )) ,'lines','highcontrast');  
	dictimg5 = SMALL_showdict(Psi_cy,[4,4], round(sqrt( K )), round(sqrt( K )) ,'lines','highcontrast');  
	dictimg6 = SMALL_showdict(Psi_y,[4,4], round(sqrt( K )), round(sqrt( K )) ,'lines','highcontrast');
	
	% Display dictionary atoms as image patch
	Hcf = figure ;
	Hca = gca ;

	subplot(3,2,1)
	imagesc(dictimg1);colormap(gray);axis off; axis image;
	title('Dict Psi\_{cxLR}')
% 	set(gca,'position',[0.05 0.6 0.4 0.3]);

	subplot(3,2,2)
	imagesc(dictimg2);colormap(gray);axis off; axis image;
	title('Dict Psi\_{xLR}')
% 	set(gca,'position',[0.55 0.6 0.4 0.3]);

	subplot(3,2,3)
	imagesc(dictimg3);colormap(gray);axis off; axis image;
	title('Dict Psi\_{cx}')
% 	set(gca,'position',[0.05 0.3 0.4 0.3]);

	subplot(3,2,4)
	imagesc(dictimg4);colormap(gray);axis off; axis image;
	title('Dict Psi\_{x}')
% 	set(gca,'position',[0.55 0.3 0.4 0.3]);

	subplot(3,2,5)
	imagesc(dictimg5);colormap(gray);axis off; axis image;
	title('Dict Psi\_{cy}')
% 	set(gca,'position',[0.05 0.05 0.4 0.3]);

	subplot(3,2,6)
	imagesc(dictimg6);colormap(gray);axis off; axis image;
	title('Dict Psi\_{y}')
% 	set(gca,'position',[0.55 0.05 0.4 0.3]);

	pause(0.02);
	
	set(gcf, 'Color', 'w'); % make the background to be white.
	FigPosition = [0 0 400 600];
	FigName = ['Dicts'];
	SetFigure_MultAxes(Hcf, Hca, FigPosition, FigFontName, FigFontSize, SaveFig, FigName, FigFormatCell) ;
end

printf('done!')
