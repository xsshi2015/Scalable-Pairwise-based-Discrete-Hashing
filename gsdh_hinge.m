function [H,W_H]=gsdh_hinge(X,options)
m=options.bit_num;
anchor_index=options.anchor_index;
[n,d]=size(X);
S=options.S;
S1=ones(size(S));
tep=find(S<0);
S1(tep)=0;
S = m/max(max(S))*S;
X=X';
[U,~,~]=svd(X*S*X(:,anchor_index)');
W_H=U(:,1:m);
clear U;
H=sign(X'*W_H);
Ha= H(anchor_index,:);
M=(X*X'+1e-3*eye(d))\X;
beta=options.beta;
Rc=options.maxIter;
bn = options.batch_num;
objB(1)=norm(H*Ha'-S,'fro');
H0=H;
for iter=1:Rc
    index=randperm(n);
    for jter=1:floor(n/bn)
        idx=index((jter-1)*bn+1:jter*bn);
        tempL = S(idx,:);
        for kter=1:m
            tempS = generateLoss(tempL, S1(idx,:),H(idx,:),Ha, kter);
            for lter=1:3           % if 3 is small, please set lter to be larger, like lter=20
                tempH=tempS*Ha(:,kter)+H(idx,kter)*beta;
                if norm(sign(tempH)-H(idx,kter), 'fro')==0
                    break;
                end
                H(idx,kter)=sign(tempH);
            end;
            Ha(:,kter) = H(anchor_index,kter);
        end;
    end;
%     objB(iter+1)=norm(H*Ha'-S,'fro');
    if norm(H-H0,'fro')==0
        break;
    else
        H0=H;
    end
end;

 for kter=1:m
    tempS = generateLoss(S, S1, H, Ha, kter);
    tempH= tempS*Ha(:,kter)+H(:,kter)*beta;
    W_H(:,kter)=M*tempH;  
end;

function L = generateLoss(S, S1, H, Ha, kter)
n= size(S,1);
m = size(H,2);
an = size(Ha,1);
Hp = H;
Hn = H;
sHp = ones(n,1);
Hp(:,kter)=sHp;
sHn = -ones(n,1);
Hn(:,kter)=sHn;

Hap = Ha;
sHap = ones(an,1);
Hap(:,kter)=sHap;

HHp = Hp*Hap';
HHn = Hn*Hap';


Lp = (0.5*S1.*(S-HHp)).^2 + (max(0.5*(ones(size(S))-S1).*(HHp), 0)).^2;
Ln = (0.5*S1.*(S-HHn)).^2 + (max(0.5*(ones(size(S))-S1).*(HHn), 0)).^2;

L = (Ln-Lp);
