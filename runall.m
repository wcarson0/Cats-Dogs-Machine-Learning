tic; %output build time in seconds

[x,y] = read_data; %Load Data
[Xtrain, ytrain, Xtest, ytest] = split_data(x,y,20); %Split Data


%Problem CP3.1
[avgcat, avgdog] = average_pet(x,y);
figure(1)
show_image(avgcat,1)
title('Average Cat')
figure(2)
show_image(avgdog,1)
title('Average Dog')


%Problem CP3.2
yguesstrain = closest_average(Xtrain,ytrain,Xtrain);
yguesstest = closest_average(Xtrain,ytrain,Xtest);
training_error = error_rate(yguesstrain,ytrain);
test_error = error_rate(yguesstest,ytest);
fprintf('Closest Average Classifier:\n')
fprintf('Closest Average training error rate is %.2g%%.\n',min(training_error));
fprintf('Closest Average test error rate is %.2g%%.\n',min(test_error));

%Problem CP3.3
yguesstrain = nearest_neighbor(Xtrain,ytrain,Xtrain);
yguesstest = nearest_neighbor(Xtrain,ytrain,Xtest);
training_error = error_rate(yguesstrain,ytrain);
test_error = error_rate(yguesstest,ytest);
fprintf('\nNearest Neighbor Classifier:\n');
fprintf('Nearest Neighbor training error rate is %.2g%%.\n',min(training_error));
fprintf('Nearest Neighbor test error rate is %.2g%%.\n',min(test_error));

%Problem CP3.4
yguesstrain = lda(Xtrain,ytrain,Xtrain);
yguesstest = lda(Xtrain,ytrain,Xtest);
training_error = error_rate(yguesstrain,ytrain);
test_error = error_rate(yguesstest,ytest);
fprintf('\nLinear Disciminant Analysis:\n');
fprintf('Linear Disciminant Analysis training error rate is %.2g%%.\n',min(training_error));
fprintf('Linear Disciminant Analysis test error rate is %.2g%%.\n',min(test_error));

%}
%Problem CP3.5
yguesstrain = perceptron(Xtrain,ytrain,Xtrain);
yguesstest = perceptron(Xtrain,ytrain,Xtest);
training_error = error_rate(yguesstrain,ytrain);
test_error = error_rate(yguesstest,ytest);
fprintf('\nPerceptron Classifier:\n');
fprintf('Perceptron training error rate is %.2g%%.\n',min(training_error));
fprintf('Perceptron test error rate is %.2g%%.\n',min(test_error));


%Problem CP3.6
pcaX = pca(Xtrain); %Determine PCA transform.
pcaXtranspose = pcaX'; %Take transpose to be in right format for show_image.
figure(3)
image_indices = [5 1 16 400]; %Change these to examine different eigenvectors.
%The code below displays a 2 x 2 grid of eigenvectors as 64 x 64 images.
for i = 1:2
    for j = 1:2
    image_index = image_indices(2*(i-1)+j);
    subplot(2,2,2*(i-1)+j)
    imagesc(reshape(pcaXtranspose(image_index,:),64,64))
    colormap('gray')
    axis square
    switch i
        case 1
            switch j
                case 1
                    content = 'Cat';
                case 2
                    content = 'Dog';
                otherwise
                    content = 'Out of Loop Bounds: Error';
            end
        case 2
            switch j
                case 1
                    content = 'Animal Eyes';
                case 2
                    content = 'Noise';
                otherwise
                    content = 'Out of Loop Bounds: Error';
            end
        otherwise
            content = 'Out of Loop Bounds: Error';
    end
            
    a = sprintf('Eigenpet %g - %s',image_index, content);
    title(a)
    end
end

fprintf('\n');

%Problem CP3.7
kvalues = [50 100 200 400];
for i = 1:length(kvalues)
    k = kvalues(i);
    
    [Xtrain_reduced, Xtest_reduced] = reduce_data(Xtrain,Xtest,k); %should result in n x k matrices 
    yguesstrain = lda(Xtrain_reduced,ytrain,Xtrain_reduced); 
    yguesstest = lda(Xtrain_reduced,ytrain,Xtest_reduced);  
    training_error = error_rate(yguesstrain,ytrain);
    test_error = error_rate(yguesstest,ytest);
    
    fprintf('For PCA-reduced data with dimension k = %g, linear discriminant analysis training error rate is %.2g%%.\n',k,training_error);
    fprintf('For PCA-reduced data with dimension k = %g, linear discriminant test error rate is %.2g%%.\n',k,test_error);
    
    yguesstrain = perceptron(Xtrain_reduced,ytrain,Xtrain_reduced);
    yguesstest = perceptron(Xtrain_reduced,ytrain,Xtest_reduced);
    training_error = error_rate(yguesstrain,ytrain);
    test_error = error_rate(yguesstest,ytest);
    
    fprintf('For PCA-reduced data with dimension k = %g, perceptron training error rate is %.2g%%.\n',k,training_error);
    fprintf('For PCA-reduced data with dimension k = %g, perceptron test error rate is %.2g%%.\n',k,test_error);
    
end


toc;
