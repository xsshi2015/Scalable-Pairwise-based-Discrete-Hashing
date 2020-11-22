function [H,W_H, objH, objB]=gsdh_ksh(X,options)
m=options.bit_num;
anchor_index=options.anchor_index;
[n,d]=size(X);
S=options.S;
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
objH(1)=0;
H0 =H;
if Rc>0
for iter=1:Rc
    index=randperm(n);
    for jter=1:floor(n/bn)
        idx=index((jter-1)*bn+1:jter*bn);
        tempS = S(idx,:);
        for kter=1:m
            for lter=1:3
                tempH=tempS*Ha(:,kter)+H(idx,kter)*beta;
                if norm(sign(tempH)-H(idx,kter), 'fro')==0
                    break;
                end
                H(idx,kter)=sign(tempH);
            end;
            Ha(:,kter) = H(anchor_index,kter);
            tempS = tempS - H(idx,kter)*Ha(:,kter)';
        end;
    end;
    objH(iter) = norm(H-H0,'fro');
    H0 =H;
    objB(iter+1)=norm(H*Ha'-S,'fro');
end;

end;

[~,Gamma,~]=svd(H'*H+1e-9*eye(m));
alpha=max(max(Gamma))+beta;
tempH=S*Ha+H*(alpha*eye(m)-(Ha'*Ha));
W_H=M*tempH;  