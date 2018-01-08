function [c,ceq] = mycon2(x)
global g_M2;
c = trace(x*x')-(g_M2);
% c = trace(x*x')-1;
ceq =[];