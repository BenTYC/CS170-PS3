% CS171   PS3
% Name: Tsung-Ying Chen 
% SID: 861310198 
% Date: 11/19/2017 
function predY = nneval(X,W1,W2)

predY = (fp(X',W1,W2))';

end
function F = fp(X,W1,W2)

A = W1 * X;
Z = 1./(1+exp(-A));
Z = [ones(1,size(Z,2));Z]; 
AO = W2 * Z;
F = 1./(1+exp(-AO));

end


