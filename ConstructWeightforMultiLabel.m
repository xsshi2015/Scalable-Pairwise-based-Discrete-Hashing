function options=ConstructWeightforMultiLabel(trainLabel,testLabel,anchor_num)
[n,c]=size(trainLabel);
[m,~]=size(testLabel);
cateTrainTest=zeros(n,m);
for iter=1:c
    index=find(trainLabel(:,iter)==1);
    idx=find(testLabel(:,iter)==1);
    cateTrainTest(index,idx)=1;
    clear index;
    clear idx;
end;
if ~isempty(anchor_num)
    index=randperm(n);
    anchor_index=index(1:anchor_num);
    S=trainLabel*trainLabel(anchor_index,:)';
    options.anchor_index=anchor_index;
else
    S=trainLabel*trainLabel';
end;
tep=find(S==0);
S(tep)=-1*floor(max(max(S))/2);
clear tep;
RankInfo=testLabel*trainLabel';
options.RankInfo=RankInfo;
options.cateTrainTest=cateTrainTest;
options.S=S;
clear RankInfo;
clear cateTrainTest;
clear S;