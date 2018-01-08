function [FinalPhi2] = CASSIdesign()
for i_s = 1:1
    
    if i_s == 1
        re_Phi_hsi_tmp =  re_Phi_hsi(1:16,:);
    else
        re_Phi_hsi_tmp =  re_Phi_hsi(17:end,:);
    end
    
    for j=1:33
        diag_gsvd(:,j) = diag(re_Phi_hsi_tmp(:,(j*16-15):j*16));
    end
    [d_row d_col]=size(diag_gsvd);

    DesCodedMask1 = [];
    DesCodedMask2 = [];
    DesCodedMask3 = [];
    re_Phi_hsi_1  = [];
    re_Phi_hsi_2  = [];
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

                distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2)+norm(abs(1-idx3),2);
                distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2)+norm(abs(idx3),2);
            elseif j == 2
                idx1 = diag_gsvd(1+4*(i-1),j+30);
                idx2 = diag_gsvd(2+4*(i-1),j+31);  
           
                distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2);
                distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2);
            elseif j == 3
                idx1 = diag_gsvd(1+4*(i-1),j+30);
                distmp1 = norm(abs(1-idx1),2);
                distmp0 = norm(abs(idx1),2);           
            end                
        
            if distmp1 > distmp0
                DesCodedMask1(j,i)=0;
            else
                DesCodedMask1(j,i)=1;
            end
        end
    end
    DesCodedMask1 = flipud(DesCodedMask1);

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
    DesCodedMask2 = flipud(DesCodedMask2);

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
                distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2)+norm(abs(1-idx3),2);
                distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2)+norm(abs(idx3),2);
            elseif j == 2
                idx1 = diag_gsvd(3+4*(i-1),j-1);
                idx2 = diag_gsvd(4+4*(i-1),j);
                distmp1 = norm(abs(1-idx1),2)+norm(abs(1-idx2),2);
                distmp0 = norm(abs(idx1),2)+norm(abs(idx2),2);
            else
                idx1 = diag_gsvd(4+4*(i-1),j-2); 
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
    DesCodedMask = [DesCodedMask1 ; DesCodedMask2 ; DesCodedMask3];

    DesCodedMask

    RescalePhi2=[];
    for i=1:33
        tempPhi2(:,:,i) = DesCodedMask(33-i+1:36-i+1,:);
        vtempPhi2 = reshape(tempPhi2(:,:,i),16,[]);
        RescalePhi2 = [RescalePhi2 diag(vtempPhi2) ];
    end
    
    
    if i_s == 1
        re_Phi_hsi_1 =  RescalePhi2;
    else
        re_Phi_hsi_2 =  RescalePhi2;
    end
    
end
FinalPhi2 = [re_Phi_hsi_1;re_Phi_hsi_2];
end