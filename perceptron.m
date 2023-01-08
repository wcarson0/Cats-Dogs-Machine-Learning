%Perceptron training error rate is 0 %
%Perceptron test error rate is 24 %

%This function takes in a training data matrix Xtrain, training
%label vector ytrain and uses them to the decision rule corresponding
%to a very simple one-layer neural network: the perceptron. 
%It also takes in a data matrix Xrun and produces a vector of label
%guesses yguess, corresponding to the sign of the linear prediction.
function yguess = perceptron(Xtrain,ytrain,Xrun)
    [ntrain, pixel] = size(Xtrain);
    ntest = size(Xrun, 1); 

    yguess = Xrun*pinv(Xtrain'*Xtrain)*Xtrain'*ytrain; 
    %ntest x 1 = nest x pixel * pixe x 1
    
   
    
    for i = 1:ntest
        if yguess(i) >= 0  %1x4096 * 4096x1 = 1x1 which is very nice
            yguess(i) = 1; 
        else
            yguess(i) = -1; 
        end
    end


end