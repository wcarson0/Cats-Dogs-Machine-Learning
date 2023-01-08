%{ 
For PCA-reduced data with dimension k = 50, linear discriminant analysis training error rate is 7.3%.
For PCA-reduced data with dimension k = 50, linear discriminant test error rate is 22%.
For PCA-reduced data with dimension k = 50, perceptron training error rate is 7.4%.
For PCA-reduced data with dimension k = 50, perceptron test error rate is 21%.
For PCA-reduced data with dimension k = 100, linear discriminant analysis training error rate is 5.8%.
For PCA-reduced data with dimension k = 100, linear discriminant test error rate is 22%.
For PCA-reduced data with dimension k = 100, perceptron training error rate is 5.8%.
For PCA-reduced data with dimension k = 100, perceptron test error rate is 23%.
For PCA-reduced data with dimension k = 200, linear discriminant analysis training error rate is 4.5%.
For PCA-reduced data with dimension k = 200, linear discriminant test error rate is 23%.
For PCA-reduced data with dimension k = 200, perceptron training error rate is 4.6%.
For PCA-reduced data with dimension k = 200, perceptron test error rate is 23%.
For PCA-reduced data with dimension k = 400, linear discriminant analysis training error rate is 2.9%.
For PCA-reduced data with dimension k = 400, linear discriminant test error rate is 27%.
For PCA-reduced data with dimension k = 400, perceptron training error rate is 2.9%.
For PCA-reduced data with dimension k = 400, perceptron test error rate is 28%.
%}


%This function takes in a training data matrix Xtrain and uses
%it to compute the PCA basis and a sample mean vector. 
%It also takes in a test data matrix Xtest and a dimension k. 
%It first centers the data matrices Xtrain and Xtest by subtracting the
%Xtrain sample mean vector from each of their rows. It then uses the 
%top-k vectors in the PCA basis to project the centered Xtrain and Xtest
%data matrices into a k-dimensional space, and outputs
%the resulting data matrices as Xtrain_reduced and Xtest_reduced.
function [Xtrain_reduced, Xtest_reduced] = reduce_data(Xtrain,Xtest,k)
    ntrain = size(Xtrain, 1); 
    ntest = size(Xtest, 1);
    
    VpcaTrain = pca(Xtrain)';          % weirdly ntrain-1 x 4096 bw
    VpcaTest = pca(Xtest)';          % weirdly ntest-1 x 4096 bw
    
    Vktrain = VpcaTrain(1:k,:)'; %4096 x k
    
    if (ntest == k)
       Vktest = [(VpcaTest(1:(k-1),:)); VpcaTest(1,:)]'; %399 X 4096 -> and appending a 1x 4096 makes it a 400 x 4096 ie k x 4096
       
    else
        Vktest = VpcaTest(1:k,:)';  %4096 x k
    end
    
    training_mean = mean(Xtrain);                % 1x4096 
    
    training_mean_vector = ones(ntrain, 1) * training_mean; %1600 x 4096
    xCenteredTrain = (Xtrain - training_mean_vector); %1600 by 4096
    
    testing_mean_vector = ones(ntest,1)*training_mean; %400*4096
    xCenteredTest = (Xtest - testing_mean_vector);  % 400*4096
    
   
    Xtrain_reduced = (xCenteredTrain*Vktrain); %should result in ntrainxk sized data - XCentered = n*4096 and Vk = 4096*k
    Xtest_reduced = (xCenteredTest*Vktest); %should result in ntrainxk sized data
    
    
    
    
end
