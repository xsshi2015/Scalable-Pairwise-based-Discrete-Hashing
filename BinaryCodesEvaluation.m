function [MAP,NDCG,Rel]=BinaryCodesEvaluation(trainH,trainLabel,testH,testLabel,RankInfo,options)
% trainH \in {-1,1}^{trn*k}, testH \in {-1,1}^{ten*k}
% RankInfo \in R^{ten*trn}

NDCG=[];
MAP=[];
Rel=[];


addpath(genpath('./evaluation'));


trainH=single(trainH>0);
testH=single(testH>0);
B = compactbit(trainH);
tB = compactbit(testH);
hammTrainTest = hammingDist(tB, B)';
[~, HammingRank]=sort(hammTrainTest,1);

TeW=options.cateTrainTest;
mRange=[500];
for iter=1:length(mRange)
    MAP01 = cat_apcal(TeW, HammingRank,mRange(iter));
    MAP(iter)=MAP01;
end;


range=[50,100];
for iter=1:length(range)
    [NDCG01,Rel01]=cal_rank(RankInfo,HammingRank',range(iter));
    NDCG(iter)=NDCG01;
    Rel(iter)=Rel01;
end;



% trainH=single(trainH>0);
% testH=single(testH>0);
% B = compactbit(trainH);
% tB = compactbit(testH);
% hammTrainTest = hammingDist(tB, B)';
% [~, HammingRank]=sort(hammTrainTest,1);
% %cateTrainTest=options.cateTrainTest;
% HR=[0,1,2,3,4,5];
% for jter=1:length(HR)
%     hammRadius=HR(jter);
%     Ret = (hammTrainTest <= hammRadius+0.00001);
%     if options.multilabel==0
%         [Pre, Rec] = evaluate_macro(RankInfo', Ret);
%         HN(jter,:)=Pre;
%         HRe(jter,:)=Rec;
%     else
%         [HN01, HRe01] = cal_rank01(RankInfo, Ret');
%         HN(jter,:)=HN01;
%         HRe(jter,:)=HRe01;
%     end;
% end;
