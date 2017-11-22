% CS171 PS3
% Name: Tsung-Ying Chen 
% SID : 861310198 
% Date: 11/19/2017 
function [W1,W2] = trainneuralnet(X,Y,nhid,lambda)

X = X';
Y = Y';
eta = 0.1;
m = size(Y,2);

W1 = randn(nhid, size(X,1)) / 10; 
W2 = randn(size(Y,1), nhid + 1) / 10;  % + 1 for offset unit

mdW = 1;
last_L = +inf;
count = 1;
while mdW >= 1e-3
    %%% Forward + Backward
    [H,F] = fp(X,W1,W2);
    [dW1, dW2] = bp(X,H,W2,F,Y);
    
    if rem(count,1000) == 0
        %%% Loss function
        regularization = (lambda/m) * ( sum(sum((W1(:,2:end)).^2)) + sum(sum((W2(:,2:end)).^2)) );
        L = ((1/m) * sum(sum((-Y .* log(F))-((1-Y) .* log(1-F))))) + regularization;
        
        %%% Adjust eta
        if L > last_L
            eta = eta / 10; 
        end
        last_L = L;
        
        %%% Display Info per 1000 Iterations
        disp(['count: ',int2str(count),' L: ',num2str(L),' mdW: ',num2str(mdW),' eta: ',num2str(eta)])
    end
    
    %%% Update
    W1_grad = (dW1 + 2 * lambda * [zeros(size(W1, 1),1) W1(:,2:end)]) ./m;
    W2_grad = (dW2 + 2 * lambda * [zeros(size(W2, 1),1) W2(:,2:end)]) ./m;
    W1 = W1 - eta * W1_grad;
    W2 = W2 - eta * W2_grad;
    
    %%% Stop Condition
    mdW = max(max(max(abs(W1_grad))), max(max(abs(W2_grad))));
    count = count + 1;
end
fprintf("\n");
end

function [H,F] = fp(X,W1,W2)

A = W1 * X;
H = 1./(1+exp(-A));
H = [ones(1,size(H,2));H]; 
AO = W2 * H;
F = 1./(1+exp(-AO));

end

function [dW1, dW2] = bp(X,H,W2,F,Y)

F_delta = F - Y;
H_delta = H .* (1-H) .* (W2'* F_delta);
H_delta(1,:) = [];     % 1st row is offset
dW1 = H_delta * X';
dW2 = F_delta * H';

end

