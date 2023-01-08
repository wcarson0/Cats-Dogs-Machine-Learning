%This function takes in a data matrix X and a label
%vector y and outputs the average cat image and average dog image.
function [avgcat, avgdog] = average_pet(X,y)
    [n, pixel] = size(X);
    
    avgcat = zeros(pixel, 1);
    avgdog = zeros(pixel, 1);
    
    ncat = 0; 
    ndog = 0; 
    
    for i = 1:n %for n examples 
        if (y(i) < 0) % if example is dog 
            ncat = ncat + 1; 
            for j = 1:pixel
                avgcat(j) = avgcat(j) + X(i, j);
            end
        else
            ndog = ndog + 1;
            for j = 1:pixel
                avgdog(j) = avgdog(j) + X(i, j);
            end
        end
    end
    
    avgcat = (avgcat./ncat)'; 
    avgdog = (avgdog./ndog)'; 
    
    
end
    
    