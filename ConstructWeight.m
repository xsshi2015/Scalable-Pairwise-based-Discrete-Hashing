function options=ConstructWeight(trainLabel,testLabel, anchor_num)
[n,d]=size(trainLabel);
[m,p]=size(testLabel);
cateTrainTest=zeros(n,m);
options.anchor_index=[];
if d==1 
    S = -1*ones(n,n);
    for iter=1:max(trainLabel)
        index=find(trainLabel==iter);
        idx=find(testLabel==iter);
        cateTrainTest(index,idx)=1;
        S(index,index)=1;
        clear index;
        clear idx;
    end;
    if ~isempty(anchor_num)
        an =floor(anchor_num/max(trainLabel));
        an_idx= [];
        count =0;
        for iter=1:max(trainLabel)
            index=find(trainLabel==iter);
            idx = randperm(length(index));
            an_idx = [an_idx; index(idx(1:an))];
            count = count+an;
        end;
        options.anchor_index = an_idx ;
        S = S(:,an_idx);
    else
      
    end;
    clear idx;
    clear index;
    RankInfo=cateTrainTest';
    options.RankInfo=RankInfo;
else
    c1=max(trainLabel(:,1));
    c2=max(trainLabel(:,2));
    train_total_Label=zeros(n,c1+c2);
    test_total_Label=zeros(m,c1+c2);
    for iter=1:c1
        index=find(trainLabel(:,1)==iter);
        train_total_Label(index,iter)=1;
        idx=find(testLabel(:,1)==iter);
        test_total_Label(idx,iter)=1;
        cateTrainTest(index,idx)=1;
    end;
    clear index;
    clear idx;
    for iter=1:c2
        index=find(trainLabel(:,2)==iter);
        train_total_Label(index,c1+iter)=1;
        idx=find(testLabel(:,2)==iter);
        test_total_Label(idx,c1+iter)=1;
        cateTrainTest(index,idx)=1;
    end;
    if isempty(anchor_num)
        S=train_total_Label*train_total_Label';
    else
        index=randperm(n);
        idx=index(1:anchor_num);
        options.anchor_index=idx;
        S=train_total_Label*train_total_Label(idx,:)';
    end;
    tep=find(S==0);
    S(tep)=-1;
    clear tep;
    RankInfo=test_total_Label*train_total_Label';
    options.trainLabel=train_total_Label;
    options.testLabel=test_total_Label;
    options.RankInfo=RankInfo;
    clear RankInfo;
    
end;
options.cateTrainTest=cateTrainTest;
options.S=S;
clear S;
clear cateTrainTest;
