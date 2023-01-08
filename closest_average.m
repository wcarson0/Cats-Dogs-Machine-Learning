%Closest average training error rate is 19 %
%Closest average test error rate is 18 %

%This function takes in a training data matrix Xtrain, training
%label vector ytrain and uses them to compute the average cat
%and dog vectors. It also takes in a data matrix Xrun and 
%produces a vector of label guesses yguess, corresponding to whether
%each row of Xtest is closer to the average cat or average dog.
function yguess = closest_average(Xtrain,ytrain,Xrun)
    [nrun, pixels] = size(Xrun);
    [avgcat, avgdog] = average_pet(Xtrain, ytrain); %each is 1x4096 ROW vector

    yguess = zeros(nrun, 1); %col vector for guess of each element in Xrun 
    
    
    dogDiff = zeros(1, pixels); %1 x 4096 
    catDiff = zeros(1, pixels); %1 x 4096 
    
    for i = 1:nrun
        dogDiff(i,:) = Xrun(i,:) - avgdog; %4000
        catDiff(i,:) = Xrun(i,:) - avgcat; %1 x 4096
    end
   
    
    distDog = zeros(nrun, 1); %ntrain x 1
    distCat = zeros(nrun, 1);
    for i = 1:nrun
        distDog(i) = norm(dogDiff(i,:));
        distCat(i) = norm(catDiff(i,:));
        if (distDog(i) < distCat(i))
            yguess(i) = 1; 
        else
            yguess(i) = -1; 
        end
    end
    
end
    
    
    
    
    
    
    