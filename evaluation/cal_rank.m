function [NDCG,REL]=cal_rank(RankInfo,IX,n)
% RankIno \in R^{ten*trn}, IX \in R^{ten*trn}
% n is the position
[ten,trn]=size(RankInfo);
tep=find(RankInfo<0);
if isempty(tep)
else
    RankInfo(tep)=0;
end;
[sortedRankInfo,~]=sort(RankInfo,2,'descend');

rel=zeros(ten,1);
ndcg=zeros(ten,1);
z=zeros(ten,1);
for iter=1:ten
    rel(iter)=1/n*sum(RankInfo(iter,IX(iter,1:n)));
    for jter=1:n
        ndcg(iter)=ndcg(iter)+(2^RankInfo(iter,IX(iter,jter))-1)/log2(jter+1);
        z(iter)=z(iter)+(2^sortedRankInfo(iter,jter)-1)/log2(jter+1);
    end;
    if z(iter)==0
        ndcg(iter)=0;
    else
        ndcg(iter)=ndcg(iter)/z(iter);
    end;
end;

REL=mean(rel);
NDCG=mean(ndcg);