%Linear discriminant analysis training error rate is 0.94 %
%Linear discriminant analysis test error rate is 21 %

%This function takes in a training data matrix Xtrain, training
%label vector ytrain and uses them to compute the cat
%and dog sample mean vectors as well as the sample covariance matrix 
%(which is assumed to be equal for cats and dogs). 
%It also takes in a data matrix Xrun and produces a vector of
%label guesses yguess, corresponding to the ML rule for
%jointly Gaussian vectors with different means and the same 
%covariance matrix.
function yguess = lda(Xtrain,ytrain,Xrun)
    [avgcat, avgdog] = average_pet(Xtrain, ytrain); 
    [ntrain, k] = size(Xtrain);
    ntest = size(Xrun, 1);
    ncat = 0;
    ndog = 0;
    avgdog = avgdog'; 
    avgcat = avgcat';
    
    for i = 1:ntrain
        if (ytrain(i) > 0)
            ndog = ndog+1; 
        else 
            ncat = ncat+1; 
        end
    end
    
    dogmatrix = zeros(ndog, k); % sample data x k 
    countdog = 1; 
    catmatrix = zeros(ncat, k); % sample data x k
    countcat = 1; 
    for i = 1:ntrain
        if (ytrain(i) > 0)
            dogmatrix(countdog,:) = Xtrain(i,:); 
            countdog = countdog + 1; 
        else 
            catmatrix(countcat,:) = Xtrain(i,:);
            countcat = countcat + 1; 
        end
    end
    
    
    sigmadog = cov(dogmatrix); %size = k x k
    sigmacat = cov(catmatrix); %size = k x k
    sigpooled = (1/(ntrain-2))*((ncat-1)*sigmacat + (ndog-1)*sigmadog); %size = k x k
    siginverse = pinv(sigpooled); %k x k 
    c = avgdog'*siginverse*avgdog - avgcat'*siginverse*avgcat; %1xk * kxk * kx1 = 1x1 
    
    bT = (2*(avgdog-avgcat)'*siginverse); % (kx1)' * kxk = 1xk
    
    yguess = zeros(ntest-1, 1); 
    
    %need yguess to be 
    xbtgc = (c <= (bT*Xrun')'); % 1x1 <=  1xk * k x ntest = 1x ntest then transpose to get vector to test error rate for  
    for i = 1:ntest
        if (xbtgc(i) == 0)
            yguess(i) = -1; 
        else 
            yguess(i) = 1; 
        end
    end
end
