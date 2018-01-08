% function [X, Recon] = GMM_CS_Inv_samePhi_optimal(y,Phi,Sig,mu,weight)
function [f] = GMM_CS_Inv_samePhi_optimal(inputPh2)
global g_y_rgb_p; global g_y_hsi_p; global g_Phi_rgb_p; global g_Sig; global g_Mu; global g_pai; global g_x_rgb_p; global g_recX;

Phi = [g_Phi_rgb_p zeros(size(g_Phi_rgb_p,1),size(inputPh2,2));...
            zeros(size(inputPh2,1), size(g_Phi_rgb_p,2))  inputPh2];
        
y = [g_y_rgb_p; g_y_hsi_p];
Sig = g_Sig;
mu  = g_Mu;
weight = g_pai;
% here y is a matrix
     [m M] = size(Phi);  % Size of the Projection Matrix
     N = size(y,2);
      K = size(mu,2); % GMM component number
      Rinv = 1e-6*eye(m); 
      % Compute the distribution  
      py = zeros(1,K);
      % Compute the mean and convariance
      for k=1:K

       P1 = inv(Phi*Sig(:,:,k)*Phi'+Rinv);
       P = (P1+P1')/2;
       %Recon.mu(:,:,k) = (Sig(:,:,k)*Phi')*P*(y-Phi*mu(:,k))+mu(:,k);
       Recon.mu(:,:,k) = bsxfun(@plus,(Sig(:,:,k)*Phi')*P*bsxfun(@minus,y,Phi*mu(:,k)),mu(:,k));
       res = bsxfun(@minus, y ,Phi*mu(:,k));
       likeli = -0.5*m*log(2*pi) + sum(log(diag(chol(P)))) - 0.5*sum(res.*(P*res),1);   
       logpy(:,k) = likeli + log(weight(k)+eps);

      end
      s = logsumexp(logpy');
      pymin = bsxfun(@minus,logpy',s);
      % sumpy = sum(py);
     
      Recon.weight = reshape(exp(pymin(:)),[K,N]); 
      
      
%       for k=1:K
%           EstX(:,k) = Recon.weight(k)*Recon.mu(:,k);
%       end
      Recon.X = squeeze(sum(bsxfun(@times, Recon.weight', shiftdim(Recon.mu,1)),2));
      X = Recon.X';
      g_recX = X;
      f =  mean(sum(abs(X(1:48,:)-g_x_rgb_p).^2));
      
end