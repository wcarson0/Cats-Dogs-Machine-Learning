%Nearest neighbor training error rate is 0%.
%Nearest neighbor test error rate is 17%.

%This function takes in a training data matrix Xtrain, training
%label vector ytrain and uses them to compute the average cat
%and dog vectors. It also takes in a data matrix Xrun and 
%produces a vector of label guesses yguess. Each guess is found
%by searching through Xtrain to find the closest row, and then 
%outputting its label.
function yguess = nearest_neighbor(Xtrain,ytrain,Xrun)
ntest = size(Xrun, 1); %test data
ntrain = size(Xtrain, 1);  %training data
 
yguess = zeros(ntest, 1);
minval = 9999;
for i = 1:ntest
   diff = zeros(ntrain, 1); 
    for j = 1:ntrain
        diff(j) = norm(Xrun(i,:)-Xtrain(j,:));
        if (diff(j) < minval)
            ival = j; 
        end
    end    
    [minval ival] = min(diff);
    yguess(i) = ytrain(ival); 
end
        %search through xtrain and find closest element and decide off that

end

