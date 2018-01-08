function X1 = Design_Mat(x_vec,m1,n1,m2,n2,m,n)
num2 = fix((m-m1)/m2)+1; nun2 = fix((n-n1)/n2)+1;
X1 = zeros(m,n); count = zeros(m,n); ct = 0;
for i = 1:m2:num2*m2
    for j = 1:n2:nun2*n2
        ct = ct + 1;
        X1(i:i+m1-1,j:j+n1-1) = X1(i:i+m1-1,j:j+n1-1) + reshape(x_vec(:,ct),[m1,n1]);
        count(i:i+m1-1,j:j+n1-1) = count(i:i+m1-1,j:j+n1-1) + 1;
    end
end
X1 = X1./(count+eps);
return