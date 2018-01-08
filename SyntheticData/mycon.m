function [c,ceq] = mycon(x)
global g_M1;
global g_M2;

% c = trace(x*pinv(x))-(g_M1);
c = trace(x*x')-g_M1;
ceq =[];