function Phi = getw_unit(Phi_P)
   
   T1 = length(Phi_P);
   T = size(Phi_P{1},3);
   
   for nt=1:T1
   temp1 = Phi_P{nt}(:,:,1);
   temp{nt,1} = diag(temp1(:));
    for nr = 2:T
        temp1 = Phi_P{nt}(:,:,nr);
        temp{nt,1} = [temp{nt,1} diag(temp1(:))];
    end
   end
    
   Phi = cell2mat(temp);
  
    
    
 