% function that generates a random covariance matrix with a specified
% trace and rank

function K = trconcovSp(nx,s,tr)

X=randn(nx,s);
K=herm(X*X');

K=tr.*K./trace(K);