load('tmpPhiGSVD_1_M32K20.mat');    

new_Phi_both = [Phi_rgb_p zeros(size(Phi_rgb_p,1),size(new_Phi_hsi,2));...
                zeros(size(new_Phi_hsi,1), size(Phi_rgb_p,2))  new_Phi_hsi];
            
    new_y_hsi_p = new_Phi_hsi*x_hsi_p;            
    new_X_rec_both = GMM_CS_Inv_samePhi([y_rgb_p;new_y_hsi_p],new_Phi_both,Sig,Mu,pai);  
    new_PSNR_joint_RGB = SS_PSNR_3D(x_rgb_p,new_X_rec_both(1:PatchSize_lar^2*rgb,:));   
    new_PSNR_joint_RGB

image_recon_joint_RGBNormDesign = patches2video_fast(new_X_rec_both(1:PatchSize_lar^2*rgb,:), Row_rgb, Col_rgb,rgb, olp_lar, olp_lar);
RECNAME ='';
RECNAME = sprintf('CASSINormDesign.png',kk);
imwrite(image_recon_joint_RGBNormDesign,RECNAME,'png');


re_Phi_hsi = new_Phi_hsi;
[rows,cols]= find(re_Phi_hsi == max((re_Phi_hsi(:))));

% re_Phi_hsi = re_Phi_hsi/re_Phi_hsi(rows,cols);
re_Phi_hsi = 10*new_Phi_hsi;

    new_Phi_both = [Phi_rgb_p zeros(size(Phi_rgb_p,1),size(new_Phi_hsi,2));...
                zeros(size(new_Phi_hsi,1), size(Phi_rgb_p,2))  re_Phi_hsi];
            
    new_y_hsi_p = re_Phi_hsi*x_hsi_p;            
    new_X_rec_both = GMM_CS_Inv_samePhi([y_rgb_p;new_y_hsi_p],new_Phi_both,Sig,Mu,pai);  
    new_PSNR_joint_RGB = SS_PSNR_3D(x_rgb_p,new_X_rec_both(1:PatchSize_lar^2*rgb,:));   
    new_PSNR_joint_RGB
    

for j=1:33
    diag_gsvd(:,j) = diag(re_Phi_hsi(:,(j*16-15):j*16));
end
[d_row d_col]=size(diag_gsvd);


for  i=1:4
    for j=1:3
        idx1 = 0;
        idx2 = 0;
        idx3 = 0;

        distmp1 = 0;
        distmp0 = 0;
        
        if j == 1
            idx1 = diag_gsvd(1+4*(i-1),j+30);
            idx2 = diag_gsvd(2+4*(i-1),j+31);
            idx3 = diag_gsvd(3+4*(i-1),j+32);  
%         1+4*(i-1)
%         j+30
%         
%         2+4*(i-1)
%         j+31
%         
%         3+4*(i-1)
%         j+32
            distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2)+norm(abs(1-idx3),2);
            distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2)+norm(abs(idx3),2);
        elseif j == 2
            idx1 = diag_gsvd(1+4*(i-1),j+30);
            idx2 = diag_gsvd(2+4*(i-1),j+31);  
%         1+4*(i-1)
%         j+30
%         
%         2+4*(i-1)
%         j+31            
            distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2);
            distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2);
        elseif j == 3
            idx1 = diag_gsvd(1+4*(i-1),j+30);
            distmp1 = norm(abs(1-idx1),2);
            distmp0 = norm(abs(idx1),2);
%         1+4*(i-1)
%         j+30            
        end
                
        distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2)+norm(abs(1-idx3),2)+norm(abs(1-idx4),2);
        distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2)+norm(abs(idx3),2)+norm(abs(idx4),2);
        
        if distmp1 > distmp0
            DesCodedMask1(j,i)=0;
        else
            DesCodedMask1(j,i)=1;
        end
    end
end
% DesCodedMask1
DesCodedMask1 = flipud(DesCodedMask1);
% DesCodedMask1

for  i=1:4
    for j=1:30
        idx1 = 0;
        idx2 = 0;
        idx3 = 0;
        idx4 = 0;
        
        idx1 = diag_gsvd(1+4*(i-1),j);
        idx2 = diag_gsvd(2+4*(i-1),j+1);
        idx3 = diag_gsvd(3+4*(i-1),j+2);  
        idx4 = diag_gsvd(4+4*(i-1),j+3);
        
        distmp1 = 0;
        distmp0 = 0;
        
        distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2)+norm(abs(1-idx3),2)+norm(abs(1-idx4),2);
        distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2)+norm(abs(idx3),2)+norm(abs(idx4),2);
        
        if distmp1 > distmp0
            DesCodedMask2(j,i)=0;
        else
            DesCodedMask2(j,i)=1;
        end
    end
end
% DesCodedMask2
DesCodedMask2 = flipud(DesCodedMask2);
% DesCodedMask2

for  i=1:4
    for j=1:3
        idx1 = 0;
        idx2 = 0;
        idx3 = 0;
        distmp1 = 0;
        distmp0 = 0;
        
        if j == 1
            idx1 = diag_gsvd(2+4*(i-1),j);
            idx2 = diag_gsvd(3+4*(i-1),j+1);
            idx3 = diag_gsvd(4+4*(i-1),j+2);
%             2+4*(i-1)
%             j
%             3+4*(i-1)
%             j+1
%             4+4*(i-1)
%             j+2
            distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2)+norm(abs(1-idx3),2);
            distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2)+norm(abs(idx3),2);
        elseif j == 2
            idx1 = diag_gsvd(3+4*(i-1),j-1);
            idx2 = diag_gsvd(4+4*(i-1),j);
%             3+4*(i-1)
%             j-1            
%             4+4*(i-1)
%             j
            distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2);
            distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2);
        else
            idx1 = diag_gsvd(4+4*(i-1),j-2); 
%             4+4*(i-1)
%             j-2
            distmp1 = norm(abs(1-idx1),2);
            distmp0 = norm(abs(idx1),2);
        end
        
        if distmp1 > distmp0
            DesCodedMask3(j,i)=0;
        else
            DesCodedMask3(j,i)=1;
        end        
    end
end
% DesCodedMask3
DesCodedMask = [DesCodedMask1 ; DesCodedMask2 ; DesCodedMask3];

DesCodedMask

RescalePhi2=[];
for i=1:33
    tempPhi2(:,:,i) = DesCodedMask(33-i+1:36-i+1,:);
    vtempPhi2 = reshape(tempPhi2(:,:,i),16,[]);
    RescalePhi2 = [RescalePhi2 diag(vtempPhi2) ];
end

    new_Phi_both = [Phi_rgb_p zeros(size(Phi_rgb_p,1),size(new_Phi_hsi,2));...
                zeros(size(new_Phi_hsi,1), size(Phi_rgb_p,2))  RescalePhi2];
            
    new_y_hsi_p = RescalePhi2*x_hsi_p;            
    new_X_rec_both = GMM_CS_Inv_samePhi([y_rgb_p;new_y_hsi_p],new_Phi_both,Sig,Mu,pai);  
    new_PSNR_joint_RGB2 = SS_PSNR_3D(x_rgb_p,new_X_rec_both(1:PatchSize_lar^2*rgb,:));   
    new_PSNR_joint_RGB2
