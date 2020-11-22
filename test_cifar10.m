clc;clear all;
bit=[8,16,32,64,128];

path='/Users/xiaoshuangshi/Dropbox/Paper/ICML2018/CIFAR10_GIST';
load([path,'/','cafData']);
load([path,'/','cafLabel']);

class_num=10;
trn_each=1e3;
ten_each=1e2;

[trainData,trainLabel,testData,testLabel]=GenerateIndex(cafData,cafLabel,trn_each,ten_each,class_num);


trainData = double(trainData);
testData = double(testData);
trainData=NormalizeFea(trainData);
testData=NormalizeFea(testData);
[trn,p]=size(trainData);
mvec = mean(trainData);
trainData = trainData-repmat(mvec,trn,1);
tn = size(testData,1);
testData = testData-repmat(mvec,tn,1);


[trn,p]=size(trainData);
m=1000;
index=randperm(trn);
anchor = trainData(index(1:m),:);
KTrain = sqdist(trainData',anchor');
sigma = mean(mean(KTrain,2));
KTrain = exp(-KTrain/(2*sigma));
mvec = mean(KTrain);
trainData = KTrain-repmat(mvec,trn,1);

tn=size(testData,1);
KTest = sqdist(testData',anchor');
KTest = exp(-KTest/(2*sigma));
testData = KTest-repmat(mvec,tn,1);

anchor_num=1000;
for iter=1:size(bit,2)
    for jter=1:5
    options = ConstructWeight(trainLabel,testLabel, anchor_num); %  options.multilabel=0;
    %options=ConstructWeightforMultiLabel(trainLabel,testLabel,anchor_num);  % options.multilabel=1;
    options.multilabel=0;
    options.bit_num=bit(iter);
    options.Range=100;
    options.anchor_num=anchor_num;
    options.sampling=0;
    options.maxIter=20;
    options.mean=0;
    options.lambda=0;
    options.beta=10;
    options.batch_num=100; % please try {20, 50, 100}
    anchor_index=options.anchor_index;
    
    
    [MAP01, objH, objB]=SDH_P(trainData,trainLabel,testData,testLabel,options);
    TM01(iter,jter,:)=MAP01;
    

    
    % Greedy method +KSH
    [MAP02,objH, objB]=GSDH_P(trainData,trainLabel,testData,testLabel,options);
    TM02(iter,jter,:)=MAP02;

    
 
    % Greedy method +BRE
    MAP03 = GSDH_P_bre(trainData,trainLabel,testData,testLabel,options);
    TM03(iter,jter,:)=MAP03;

    
    
    %Greedy method + Hinge loss
    MAP04 = GSDH_P_hinge(trainData,trainLabel,testData,testLabel,options);
    TM04(iter,jter,:)=MAP04;

    end;

end;