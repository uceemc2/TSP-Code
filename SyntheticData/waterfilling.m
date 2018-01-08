% water filling algorithm

function [c Q]=waterfilling(H,P,d)
if d==1
    s=abs(diag(H));
else
    s=sqrt(eig(H'*H));
end
lambda=1./abs(s).^2;
u = (P+sum(lambda))/length(lambda);
Q = u-lambda; Q = Q.*(Q > 0);
Pu = sum(Q);
while (Pu > P*(1+1e-4)),
    u = u-(Pu-P)/length(Q);
    Q = u-lambda; Q = Q.*(Q > 0);
    Pu = sum(Q);
end
v=log2(u./lambda);
v=v.*(v>0);
c=sum(v);
