function [Pre,Rec]=evaluate(trainLabel,TeW,IX,options)

[numtrain, numtest] = size(IX);
c=size(trainLabel,2);
RNN=options.Range;
pr=[];
re=[];
for i = 1 : numtest
    y = IX(:,i);

    new_label=TeW(:,i)';
    for k=1:size(RNN,2)
    x=0;
    p=0;
    num_return_NN = RNN(k);%5000; % only compute MAP on returned top 5000 neighbours.
    for j=1:num_return_NN
        if new_label(y(j))==1
            x=x+1;
        end
    end  
    pr(i,k)=x/num_return_NN;
    re(i,k)=c*x/numtrain;
    end;
    
    
end

Pre = mean(pr,1);
Rec=mean(re,1);

% figure
% plot(Rec, Pre, 'b->','LineWidth',2);
% axis([0 1 0 1])
% xlabel('recall')
% ylabel('precision')
