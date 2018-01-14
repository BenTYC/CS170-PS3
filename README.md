# CS171-Neural-Network

Implement a two-layer (2 layers of weights, 1 hidden layer of units) neural network for classification with all non-linearities as sigmoids. The file usps_all.mat contains the data for hand-written digits. The supplied function getusps will load them in and produce a training set and a testing set for distinguishing the “7s” from the “9s.”1 In this case the Y values are either 0 or 1 (instead of -1 or +1), because that matches a sigmoid output more naturally.  
  
Produce a plot of the testing error rate as a function of   (the regularization strength) for three different numbers of hidden units: 5, 10, and 50. The supplied code runusps does this.  
  
It calls two functions you should write: [W1,W2] = trainneuralnet(X,Y,nhid,lambda) and predY = nneval(X,W1,W2). Getting a neural network to converge can be a little tricky. Please follow the guide- lines below.  
  

1. Start the weights randomly chosen using randn (that is each weight is selected from a normal distri- bution with mean 0 
and unit standard deviation). Then divide all weights by 10 to make them closer to 0.  
2. Each layer should have an “offset” unit (to supply a 1 to the next layer), except the output layer.  
3. For a problem this small, use batch updates. That is, the step is based on the sum of the gradients for each element in the training set.  
4. For the step size start with eta = 0.1.  
5. For real neural network training, people use rules like those discussed on http://sebastianruder.com/optimizing-gradient-descent/index.html, but they are more complex than needed for this simple example. Instead, every 1000 iterations, check to see if the loss function (including the regularization part) has decreased over the past 1000 iterations. If it has not, divide ⌘ by 10 and continue.
6. Check the maximum gradient element (before multiplying it by eta). If its absolute value is less than 10^-3, stop the algorithm.  

