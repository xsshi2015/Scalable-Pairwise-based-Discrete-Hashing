function MAP=GSDH_P_hinge(trainData,trainLabel,testData,testLabel,options)
MAP=[];
NDCG= [];
REL= [];



[H,W_H]=gsdh_hinge(trainData,options);
trainB=single(H>0);
testH=single(testData*W_H>0);


if options.multilabel==0
    RankInfo = options.cateTrainTest';
    [MAP,NDCG,REL]=BinaryCodesEvaluation(trainB,trainLabel,testH,testLabel,RankInfo,options);
else
    NDCG=[];
    REL=[];
    RankInfo=options.RankInfo;
    [MAP,NDCG,REL]=BinaryCodesEvaluation(trainB,trainLabel,testH,testLabel,RankInfo,options);
end