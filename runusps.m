function runusps

[trainX,trainY,testX,testY] = getusps(7,9,550);

nhiddens = [5 10 50];
lambdas = logspace(-4,0,5);
lambdas = lambdas*length(trainY);
erates = nan(length(lambdas),length(nhiddens));
%{
fprintf('trainX: %d x %d\n', size(trainX, 1), size(trainX, 2));
fprintf('trainY: %d x %d\n', size(trainY, 1), size(trainY, 2));
fprintf('testX: %d x %d\n', size(testX, 1), size(testX, 2));
fprintf('testY: %d x %d\n\n', size(testY, 1), size(testY, 2));
%}
li = 1;
for lambda=lambdas
	ni = 1;
	for nhidden=nhiddens
		mingrad = 1e-3;
		% this function (trainneuralnet) you are to supply!
		[W1,W2] = trainneuralnet(trainX,trainY,nhidden,lambda);
		% this function (nneval) you are to supply too!
		predY = nneval(testX,W1,W2);
		predY(predY<0.5) = 0;
		predY(predY>=0.5) = 1;
		testerr = sum(predY~=testY)/length(testY);
		erates(li,ni) = testerr;
		plotit(lambdas,nhiddens,erates);
		ni=ni+1;
    end;
	li=li+1;
end;

figure(1);
print -dpdf result.pdf;


function plotit(ls,ns,errs)

figure(1);
hold off;
ll = cell(length(ns),1);
for i=1:length(ns)
     loglog(ls,errs(:,i));
     hold on;
     ll{i} = num2str(ns(i));
end;
legend(ll{:})
xlabel('lambda');
ylabel('testing error rate');
hold off;
drawnow;
