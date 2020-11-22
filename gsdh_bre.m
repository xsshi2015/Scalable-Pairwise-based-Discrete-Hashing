function [H,W_H]=gsdh_bre(X,options)
m=options.bit_num;
anchor_index=options.anchor_index;
[n,d]=size(X);
S=options.S;
tep=find(S<0);
S(tep)=0;
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
H0 =H;
for iter=1:Rc
    index=randperm(n);
    for jter=1:floor(n/bn)
        idx=index((jter-1)*bn+1:jter*bn);
        tempS = S(idx,:);
        for kter=1:m
            tempL = generateLoss(tempS, H(idx,:), Ha, kter);
            for lter=1:10
                tempH=tempL*Ha(:,kter)+H(idx,kter)*beta;
                if norm(sign(tempH)-H(idx,kter), 'fro')==0
                    break;
                end
                H(idx,kter)=sign(tempH);
            end;
            Ha(:,kter) = H(anchor_index,kter);
        end;
    end;
    objB(iter+1)=norm(H*Ha'-S,'fro');
    if norm(H-H0,'fro')==0
        break;
    else
        H0=H;
    end
end;

[~,Gamma,~]=svd(H'*H+1e-9*eye(m));
alpha=max(max(Gamma))+beta;
tempH=S*Ha+H*(alpha*eye(m)-(Ha'*Ha));
W_H=M*tempH;  


function L = generateLoss(S, H, Ha, kter)
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


Lp = (S-HHp).^2; 
Ln = (S-HHn).^2; 

L = (Ln-Lp);
