function [MAP,objH,objB]=GSDH_P(trainData,trainLabel,testData,testLabel,options)

MAP=[];
NDCG=[];
Rel=[];


[H,W_H, objH, objB]=gsdh_ksh(trainData,options);
trainB=single(H>0);
testH=single(testData*W_H>0);


if options.multilabel==0
    RankInfo = options.cateTrainTest';
    [MAP, NDCG, Rel]=BinaryCodesEvaluation(trainB,trainLabel,testH,testLabel,RankInfo,options);
else
    RankInfo=options.RankInfo;
    [MAP, NDCG, Rel]=BinaryCodesEvaluation(trainB,trainLabel,testH,testLabel,RankInfo,options);
end