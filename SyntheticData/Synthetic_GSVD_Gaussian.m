clear all;
clc; 
close all;

global g_M1;
global g_M2;
global g_Phi2;
global g_Sigx1x2;
global g_Ri;
global g_mux1x2;
global g_pcvett;
global g_data1;
global g_w;
global g_nx1;
global g_nx2;
global g_Phi1;

global g_pcc;
global g_newmux;
global g_newsig;
global g_Ri1;
global g_w1;

options = optimoptions('fmincon','MaxIter',8000,'MaxFunEvals',240000)
np= 15; vettSNRdB=logspace(-1,6,np);

nx1= 14;
nx2=  6; 
NumReal=2e3;

K1=1; K2=1;

pcvett=rand(K1*K2,1);
pcvett=pcvett./sum(pcvett);
pcjointmat=reshape(pcvett,K2,K1);
pc1mat=repmat(sum(pcjointmat),K2,1);
pc2mat=repmat(sum(pcjointmat),K1,1);

scvett=[2 2 2 2];
s1vett=[2 2 2 2];
s2vett=[1 1 1 1];

vettM1=1:4;
M2 = 2;

Pc1mat=zeros(nx1,max(scvett),K1*K2);
Pc2mat=zeros(nx2,max(scvett),K1*K2);
P1mat=zeros(nx1,max(s1vett),K1*K2);
P2mat=zeros(nx2,max(s2vett),K1*K2);
Sigx1x2=zeros(nx1+nx2,nx1+nx2,K1*K2);
Sigx1=zeros(nx1,nx1,K1*K2);
In=randn(max(nx1,nx2));
In2=randn(max(nx1,nx2));

for cc=1:(K1*K2)
    Pc1mat(:,1:scvett(cc),cc)= In(1:nx1,1+cc:scvett(cc)+cc);
    Pc2mat(:,1:scvett(cc),cc)=In2(1:nx2,1+cc:scvett(2)+cc);
    P1mat (:,1:s1vett(cc),cc)= In(1:nx1,(scvett(cc)+1+cc:scvett(cc)+s1vett(cc)+cc));
    P2mat (:,1:s2vett(cc),cc)=In2(1:nx2,(scvett(cc)+s1vett(cc)+1+cc:scvett(cc)+s1vett(cc)+s2vett(cc)+cc));
    tSig=[Pc1mat(:,1:scvett(cc),cc) P1mat(:,1:s1vett(cc),cc) zeros(nx1,s2vett(cc)); Pc2mat(:,1:scvett(cc),cc) zeros(nx2, s1vett(cc)) P2mat(:,1:s2vett(cc),cc)];
    Sigx1x2(:,:,cc)=herm(tSig*tSig');
    Sigx1(:,:,cc) =Sigx1x2(1:nx1,1:nx1,cc);
    Sigx2(:,:,cc) =Sigx1x2(nx1+1:nx1+nx2,nx1+1:nx1+nx2,cc);
    Sigx12(:,:,cc)=Sigx1x2(1:nx1,nx1+1:nx1+nx2,cc);
    Sigx21(:,:,cc)=Sigx1x2(nx1+1:end,1:nx1,cc);
end
for cc=1:(K1*K2)
    rx1=rank(Sigx1(:,:,cc))
    rx2=rank(Sigx2(:,:,cc))
    rx=rank(Sigx1x2(:,:,cc))
end

% input mean vectors
mux1x2=zeros(nx1+nx2,K1*K2);
mux1x2=zeros(nx1+nx2,K1*K2);
mux1=mux1x2(1:nx1,:);
mux2=mux1x2(nx1+1:end,:);

% Initialization of the measurement kernels
Phi01=randn(vettM1(end),nx1);
Phi02=randn(M2,nx2);
    
% input source
[p,c] = size(mux1x2);
X = zeros(p,NumReal);
label = randsample(1:c,NumReal,true,pcvett);
for t = 1:c
    rg = find(label==t); 
    len = length(rg);
    if len>0, X(:,rg) = mux1x2(:,t)*ones(1,len) + herm(sqrtm(Sigx1x2(:,:,t)))'*randn(p,len); end
end
data1 = X;
x1 = data1(1:nx1,:);
x2 = data1(nx1+1:end,:);

for kkk=1:K1*K2      
            AA = (Sigx2(:,:,kkk));
            BB = (Sigx2(:,:,kkk)-Sigx21(:,:,kkk)*pinv(Sigx1(:,:,kkk))*Sigx12(:,:,kkk));                      
            A = herm(sqrtm(herm(AA)));
            B = herm(sqrtm(herm(BB)));                       

            [U,V,X,C,S] = gsvd(A,B,0);
            rx1 = rank(Sigx1(:,:,kkk));
            rx2 = rank(Sigx2(:,:,kkk));
            rx  = rank(Sigx1x2(:,:,kkk));
            tmpR= rx1+rx2-rx;
            tmpC= nx2 - tmpR;
            tmpPhi2(:,:,kkk) = [zeros(tmpR,tmpC) eye(tmpR)]*inv(X);
end
DesPhi2  = cat(1,tmpPhi2(1,:,1),tmpPhi2(2,:,1));
testPhi2 = cat(1,DesPhi2(1,:), DesPhi2(2,:));
itotal = 1;

for nn=1:length(vettM1)

    M1=vettM1(nn);
    P1=M1;     
    P2=M2;

    Phi1=Phi01(1:M1,:);
    Phi1=sqrt(P1)*Phi1./trace(herm(Phi1*Phi1'));
    Phi2=Phi02(1:M2,:);
    Phi2=sqrt(P2)*Phi2./trace(herm(Phi2*Phi2'));    
    Phi=[Phi1 zeros(M1,nx2); zeros(M2,nx1) Phi2];
       
    DesPhi = zeros(M1+M2,nx1+nx2);
    
for i = 1:itotal

    DesPhi(:,:,i)=[Phi1 zeros(M1,nx2); zeros(M2,nx1) testPhi2(:,:,i)];        
    for k = 1:np
        snr=10.^(10*log10(vettSNRdB(k))/10);
        rho=1/sqrt(snr);
        Ri1= rho^2*eye(M1);
        Ri2= rho^2*eye(M2);
        Ri  = [Ri1 zeros(M1,M2); zeros(M2,M1) Ri2];
        w1 = rho*randn(M1,NumReal);
        w2 = rho*randn(M2,NumReal);
        w  = [w1;w2];

        y = Phi*data1 + w;
        xhat_dist = GMMcondMeanEstColorNoise(y,Sigx1x2,mux1x2,pcvett,Phi,Ri);
        mse_dist(k,nn)=mean(sum(abs(xhat_dist-data1).^2));    
        xhat_si=xhat_dist(1:nx1,:);
        mse_SI_p1_p2(k,nn)=mean(sum(abs(xhat_si-data1(1:nx1,:)).^2));
        mse_SI_p1_p2(k,nn)     
          
        Desy = DesPhi(:,:,i)*data1 + w;        
        RX = GMMcondMeanEstColorNoise(Desy,Sigx1x2,mux1x2,pcvett,DesPhi(:,:,i),Ri);       
        mse_dec_optphi2(k,nn,i) = mean(sum(abs(RX(1:nx1,:)-data1(1:nx1,:)).^2));
        mse_dec_optphi2(k,nn,i)
        
    if i == 1
        g_M1     = M1;
        g_M2     = M2;
        g_Phi1   = Phi1;
        g_Sigx1x2= Sigx1x2;
        g_Ri     = Ri;
        g_mux1x2 = mux1x2;
        g_pcvett = pcvett;
        g_data1  = data1;
        g_w      = w;
        g_nx1    = nx1;
        g_nx2    = nx2;

        optPhi2 = sym('optPhi2',[size(Phi2)]);
        x0  = Phi2;
        A=[]; b=[]; Aeq=[]; beq=[]; lb=[]; ub=[];
        [c,ceq] = mycon2(optPhi2);
        redoflag = 0; icount = 0; tmpfval =0;
        while redoflag<=0
            icount = icount+1;
            [optPhi2,fval,exitflag,output] = fmincon(@myfunmulti_phi2,x0,A,b,Aeq,beq,lb,ub,@mycon2,options);
            redoflag = exitflag;
            tmpfval(icount) = fval;
            if icount>3            
                fval = min(tmpfval);
                break;
            end
            if redoflag <=0     
                x0=randn(M2,nx2);   
                x0=x0(1:M2,:);
                x0=sqrt(P2)*x0./trace(herm(x0*x0'));             
            end
        
        end        
        mse_dec_optphi2_numercial(k,nn) = fval;
        mse_dec_optphi2_numercial(k,nn)
    end
      
end
    end
end
for i = 1: itotal
    figure(i)
    plot(10*log10(vettSNRdB'),10*log10(mse_SI_p1_p2),'-','LineWidth',2)
    grid on
    xlabel('1/\sigma^2 (dB)')
    ylabel('MMSE (dB)') 
    hold on
    plot(10*log10(vettSNRdB'),10*log10(mse_dec_optphi2(:,:,i)),'--o','LineWidth',1)
    plot(10*log10(vettSNRdB'),10*log10(mse_dec_optphi2_numercial(:,:)),'-->','LineWidth',1)
end