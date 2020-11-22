function [H,W_H, objH, objB]=sdh_ksh(X,options)
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
H0=H;
if Rc>0
for iter=1:Rc
    index=randperm(n);
    for jter=1:floor(n/bn)
        idx=index((jter-1)*bn+1:jter*bn);
        [~,Gamma,~]=svd(Ha'*Ha+1e-9*eye(m));
        alpha=max(max(Gamma))+beta;
        for lter=1:10
            tempH=S(idx,:)*Ha+H(idx,:)*(alpha*eye(m)-Ha'*Ha);
            sH=sign(tempH);
            if norm(H(idx,:)-sH,'fro')==0
               break; 
            end
            H(idx,:)=sH;
        end;
        Ha = H(anchor_index,:);
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