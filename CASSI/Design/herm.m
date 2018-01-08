% hermitian part of a matrix
function y=herm(x)
y=(x+x')/2;
r=norm(y-x,'fro').^2/norm(x,'fro').^2;
if r>1e-9
    disp(['Problems with hermitian part of the matrix. ratio = ',num2str(r)])
    pause
end