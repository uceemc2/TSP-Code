

% load RGB
filename = 'scene1';
mea_HSI = 1;
mea_rgb = 'rgb';

Row_hsi = 256;
Col_hsi = 256;
PatchSize = 4;

Row_rgb = 512;
Col_rgb = 512;
PatchSize_lar = 8;

addpath(filename);


list_rgb = dir([filename '/*.bmp']);
x_rgb0 = im2double(imread(list_rgb.name));
clear list_rgb

% load the HSI data
mea_HSI = 1;
mea_rgb = 'rgb';
resize_flag = 1;
Row_hsi = 256;
Col_hsi = 256;
PatchSize = 4;


list_hsi = dir([filename '/*_reg1.mat']);
load(list_hsi.name);
x_hsi0 = reflectances;
clear reflectances list_hsi

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
Row_new_rgb = floor(Row/PatchSize_lar)*PatchSize_lar;
Col_new_rgb = floor(Col/PatchSize_lar)*PatchSize_lar;
x_rgb = x_rgb(1:Row_new_rgb, 1:Col_new_rgb,:);

% ******************* Signal of Interest (x1,Phi1,y1)*******************
% 1) Generate Phi1 (for compressing RGB)
Cr=0.2126;
Cg=0.7152;
Cb=0.0722;

Phi_Cr = Cr*eye(PatchSize_lar*PatchSize_lar);
Phi_Cg = Cg*eye(PatchSize_lar*PatchSize_lar);
Phi_Cb = Cb*eye(PatchSize_lar*PatchSize_lar);

Phi_rgb_p = [Phi_Cr Phi_Cg Phi_Cb];

% 2) Vectorized 3 channel RGB (512*512*3) then using 1 channel RGB 192*4096
olp_lar = PatchSize_lar;
x_rgb_p = video2patches_fast(x_rgb,PatchSize_lar, PatchSize_lar,olp_lar,olp_lar);  % Here you can use 'sliding'
% 3) Generate grayscale patches
y_rgb_p = Phi_rgb_p*x_rgb_p;


% ******************* Side Information (x2,Phi2,y2) *******************
% generate the Phi_matrix with the CASSI scenario
T = mea_HSI;
for t= 1:T
Phi{t} = binornd(1,0.5,[Row_new, Col_new,Ch]);
end
shift = 1;
shift_T = 17;
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

Phi_hsi   = getw_unit(Phi_new_unit); %y_hsi_p_check = Phi_hsi*x_hsi_p;

% now y is the measurement of the HSI image
for t1 = 1:T 
    y_hsi{t1} = sum(x_hsi.*Phi_new{t1},3);
end


olp = PatchSize; % here we use non-overlapping patches
% vectorize the patches
y_hsi_all = reshape(cell2mat(y_hsi),[Row_new, Col_new,T]);
% 2) Vectorized 33 channels (256*256*33) then using 528*4096 (x_hsi_p)
x_hsi_p = video2patches_fast(x_hsi,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'
% 3) Generate single snapshot image (y_hsi_p)
y_hsi_p = video2patches_fast(y_hsi_all,PatchSize, PatchSize,olp,olp);  % Here you can use 'sliding'

% Generate Phi2 (CASSI projection kernel for compressing HSI)


