%This function takes in a vector of true labels ytrue
%and a vector of guessed labels yguess and reports back
%the error rate of the guesses as a percentage 0% to 100%.
function err = error_rate(ytrue,yguess)

n = length(ytrue);
err = 100/n*sum([yguess ~= ytrue]);
