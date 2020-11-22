function ap = cat_apcal(TeW, IX, num_return_NN)
% ap=apcal(score,label)
% average precision (AP) calculation 
Acc=[];
[numtrain, numtest] = size(IX);
apall = zeros(1,numtest);
count=0;
for i = 1 : numtest
    y = IX(:,i);
    x=0;
    p=0;
    new_label=TeW(:,i)';
    
    %num_return_NN =floor(500);%5000; % only compute MAP on returned top 5000 neighbours.
    for j=1:num_return_NN
        if new_label(y(j))==1
            x=x+1;
            p=p+x/j;
        end
    end  
    if p==0
        apall(i)=0;
    else
        apall(i)=p/x;
    end
end
ap = mean(apall);