clc
clear

%% Training an ANN

close all
load('NFkB.mat')
Y_Exp=Y;
clear Y
TNF_dose=[.5 1 2];
dose=3;
error_idx=4;
t_sam=0:1:14;
tt=0:0.1:14;
w0=zeros(dose*(length(t_sam))*length(error_idx),1);

%% without error terms
load('Init.mat')

options = odeset('AbsTol',1e-6,'RelTol',1e-6);
dY=[];

mak1={'bo','ro','ko'};
mak2={'b-','r-','k-'};

for i=1:length(TNF_dose)
    u=TNF_dose(i);    
 
    R1=ode15s(@(t,x) case_det(t,x,u),tt,x0,options);    
    y=deval(R1,tt);   
    Y(:,:,i)=y;
     R1=ode15s(@(t,x) case_det(t,x,u),t_sam,x0,options);    
    y=deval(R1,t_sam);   
    for j=1:length(t_sam)
        dy(:,j)=case_det(t_sam,y(:,j),u);
   end
   dY=[dY dy];
end
DYmax=max(abs(dY),[],2);
for i=1:dose
    DDx(:,:,i)=repmat(DYmax(error_idx),1,length(t_sam));
     
end 
Alpha=[  1e-3 1e-3;
        1e-3 1e-2;
        1e-3  0.1;
        1e-3  1;
        1e-3  10]; 

Ntrial=10;
load('Cross_validation.mat')
Tr_IDX=IDX{1};
Val_IDX=IDX{2};

Tr_IDX=repmat(Tr_IDX,Ntrial,1);
Val_IDX=repmat(Val_IDX,Ntrial,1);
%%
load('Res.mat')
XXX=Res{1};
SSS=Res{2};

tt_idx=1:2:20;
vv_idx=2:2:20;
for i=1:size(Alpha,1)
    s=[];
   for j=1:Ntrial
      St=squeeze(SSS(j,:,:,i))'; 
      St=St(:,1:2);
      s=[s St];
   end
   S(:,1,i)=min(s(:,tt_idx),[],2);
   S(:,2,i)=min(s(:,vv_idx),[],2);
end
clear SS s SS
for i=1:5
   SS(i,:)=mean(S(:,:,i));      
end
SS=[SS SS(:,1)/size(Tr_IDX,2)+SS(:,2)/size(Val_IDX,2)];
Alpha_idx=find(SS(:,3)==min(SS(:,3)));
alpha=Alpha(Alpha_idx,2);
SS_op=S(:,:,Alpha_idx);
for i=1:5
   St=squeeze(SSS(:,:,i,Alpha_idx)); 
    St(:,3)=St(:,1)/size(Tr_IDX,2)+St(:,2)/size(Val_IDX,2);
    idx(i,1)=find(St(:,3)==min(St(:,3)));
    idx(i,2)=min(St(:,3));
end
for i=1:5
    x_opt(i,:)=XXX(idx(i,1),:,i,Alpha_idx);
end
X_opt=x_opt(1,:);

%%

XS=[];
 W=reshape(X_opt,length(error_idx),length(t_sam),length(TNF_dose));
 W=W.*DDx;
 WS=reshape(W,1,length(error_idx)*length(t_sam)*length(TNF_dose));
for i=1:3
  XS=[XS [Y(:,:,i);tt;repmat(TNF_dose(i),1,length(tt))]]; 
  
end

WS=[];
for i=1:3

     E=W(:,:,i);
     pp= interp1(t_sam,E,'linear','pp');
     e=ppval(pp,tt);
    WS=[WS e];
end

Ncan=1:10;
Ntr=300;
Res=zeros(Ntr,4,length(Ncan)+1,length(Ncan));
AIC_idx=[];
for i=1:length(t_sam)
    AIC_idx=[AIC_idx find(tt==t_sam(i))];
end
%% optimizing the structure of an ANN
AIC_idx=[AIC_idx AIC_idx+length(tt) AIC_idx+2*length(tt)];
non_AICidx=setdiff(1:length(WS),AIC_idx);
parfor i=1:10%
    Temp=zeros(Ntr,4,length(Ncan)+1);
    for k=0:10
        temp=zeros(Ntr,4);
        for j=1:Ntr
            rng('shuffle')
            trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
            if k==0
                 net = feedforwardnet( i,trainFcn);
            else     
                 net = feedforwardnet( [i,k],trainFcn);
            end
            net = configure(net,XS,WS);
            net.layers{1}.transferFcn='tansig';
            net.divideFcn='divideind';
            net.trainParam.showWindow = false;
            
            AICtemp_idx=AIC_idx; 
            AICidx_trn=datasample(AIC_idx,floor(0.7*length(AIC_idx)),'Replace',false);
            AICtemp_idx=setdiff(AICtemp_idx,AICidx_trn);
            AICidx_val=datasample(AICtemp_idx,floor(0.15*length(AIC_idx)),'Replace',false);
            AICtemp_idx=setdiff(AICtemp_idx,AICidx_val);
            AICidx_test=AICtemp_idx;
            
             rng('shuffle')
            nAICtemp_idx=non_AICidx; 
            nAICidx_trn=datasample(non_AICidx,floor(0.7*length(non_AICidx)),'Replace',false);
            nAICtemp_idx=setdiff(nAICtemp_idx,nAICidx_trn);
            nAICidx_val=datasample(nAICtemp_idx,floor(0.15*length(non_AICidx)),'Replace',false);
            nAICtemp_idx=setdiff(nAICtemp_idx,nAICidx_val);
            nAICidx_test=nAICtemp_idx;
                
            net.divideParam.testInd=[AICidx_test nAICidx_test];
            net.divideParam.valInd=[AICidx_val nAICidx_val];
             net.divideParam.trainInd=[AICidx_trn nAICidx_trn];
            
            [net,tr]=train(net,XS,WS);
            wp=net(XS);
            n = length(tr.trainInd);
            p = length(getwb(net));
            AIC=n*log(sum((WS( tr.trainInd)-wp( tr.trainInd)).^2)/n)+2*p+2*p*(p+1)/(n-p-1);
            
            temp(j,1)=AIC;
            AIC=n*log(sum((WS-wp).^2)/n)+2*p+2*p*(p+1)/(n-p-1);
            temp(j,2)=AIC;
            n=length(AICidx_trn);
            AIC=n*log(sum((WS(AICidx_trn)-wp(AICidx_trn)).^2)/n)+2*p+2*p*(p+1)/(n-p-1);
            temp(j,3)=AIC;
            
            AIC=n*log(sum((WS(AIC_idx)-wp(AIC_idx)).^2)/n)+2*p+2*p*(p+1)/(n-p-1);
            temp(j,4)=AIC;

        end
        k1=k+1;
        Temp(:,:,k1)=temp;
    end
    
    Res(:,:,:,i)=Temp;
end
save('Res.mat')
