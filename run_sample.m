function run_sample

X = [1 2 -1];
Y = 1;
nhid = 3;

[dW1,dW2,dW3] = train(X,Y);

disp(dW1)
disp(dW2)
disp(dW3)

end

function [dW1,dW2,dW3] = train(X,Y)
X = X';
Y = Y';
W1 = [-1 2 2;0 -1 3;1 5 0]; 
W2 = [0 0 1 2; -2 1 -1 1];  
W3 = [4 -5 2]; 

[Z,Z2,F] = fp(X,W1,W2,W3);
[dW1, dW2,dW3] = bp(X,Z,Z2,F-Y,W2,W3);
end

function [Z,Z2,F] = fp(X,W1,W2,W3)

A = W1 * X;
Z = 1./(1+exp(-A));
Z = [ones(1,size(Z,2));Z]; 
A2 = W2 * Z;
Z2 = 1./(1+exp(-A2));
Z2 = [ones(1,size(Z2,2));Z2]; 
A3 = W3 * Z2;
F = 1./(1+exp(-A3));

end

function [dW1, dW2, dW3] =bp(X,Z,Z2,dY,W2,W3)

delta2 = Z2 .* (1-Z2) .* (W3'*dY);
delta2(1,:) = [];     % 1st row is offset
delta1 = Z .* (1-Z) .* (W2'*delta2);
delta1(1,:) = [];     % 1st row is offset
dW3 = dY * Z2';
dW2 = delta2 * Z';
dW1 = delta1 * X';

end
