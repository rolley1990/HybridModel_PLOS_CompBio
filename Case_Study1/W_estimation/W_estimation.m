%% this file is to estimate the correction terms (W)

clc
clear

TNF_dose=[.5 1 2];
dose=3;
error_idx=4;
t_sam=0:1:14;

w0=zeros(dose*(length(t_sam))*length(error_idx),1);

%% without error terms
load('Init.mat')
load('NFkB.mat')
Y_Exp=Y;
clear Y
options = odeset('AbsTol',1e-6,'RelTol',1e-6);
dY=[];
for i=1:length(TNF_dose)
    u=TNF_dose(i);
    
    R1=ode15s(@(t,x) case_det(t,x,u),t_sam,x0,options);
    
    y=deval(R1,t_sam);
    
    Y(i,:)=y(3,:);
   for j=1:length(t_sam)
        dy(:,j)=case_det(t_sam,y(:,j),u);
   end
   dY=[dY dy];
end

DYmax=max(abs(dY),[],2);

%%

Ntrial=10;
WW0=[];
for i=1:Ntrial
  temp=rand(length(w0),1)*200-100;
  WW0=[WW0 repmat(temp,1,5)];
end
load('Cross_validation.mat')
Tr_IDX=IDX{1};
Val_IDX=IDX{2};

Tr_IDX=repmat(Tr_IDX,Ntrial,1);
Val_IDX=repmat(Val_IDX,Ntrial,1);


for i=1:dose
    DDx(:,:,i)=repmat(DYmax(error_idx),1,length(t_sam));
end 

UB=100*ones(length(w0),1);
LB=-100*ones(length(w0),1);

rng('shuffle')

Alpha=[  1e-3 1e-3;
        1e-3 1e-2;
        1e-3  0.1;
        1e-3  1;
        1e-3  10]; 

tic;
opts=optimoptions(@fmincon,'MaxFunEvals',20000,'MaxIterations',20000,'Disp','final','FinDiffType','central');

XXX=zeros(Ntrial,length(w0),5,5);  % Ntrial, w0size, five-fold crossvalidation, size of Alpha
SSS=zeros(Ntrial,3,5,5);


for k=1:size(Alpha,1)
    
    X=zeros(5*Ntrial ,length(w0));
    F=zeros(5*Ntrial ,1);
    VF=F;
    EF=F;
    
    alpha1=Alpha(k,1);
    alpha2=Alpha(k, 2);
    
    parfor jj=1:5*Ntrial
              
        j=jj;
        i= jj;
        
        tr_idx=Tr_IDX(i,:);
        vl_idx=Val_IDX(i,:);
        fit=@(w) fitfcn(w,error_idx,alpha1,alpha2,DDx,tr_idx,[],Y_Exp,x0);
            
        exitflag=0;
        W0=WW0(:,jj);
        while exitflag==0
            [x,f,exitflag,~]=fmincon(fit,W0,[],[],[],[],LB,UB,[],opts);
            W0=x;
        end     
        X(jj,:)=x;
        EF(jj)=exitflag;
        f
        [F(jj),VF(jj)]=fitfcn(x,error_idx,alpha1,alpha2,DDx,tr_idx,vl_idx,Y_Exp,x0);
        
    end
    
    XX=zeros(Ntrial,length(w0),5);
    S=[F VF EF];
    for i=1:Ntrial
        for j=1:5
            ii=5*(i-1)+j;
            XX(i,:,j)=X(ii,:);
            SS(i,:,j)=S(ii,:);
        end
    end
    
    XXX(:,:,:,k)=XX;  % Ntrial, w0size, five-fold crossvalidation, size of Alpha
    SSS(:,:,:,k)=SS;
    
end

Res{1}=XXX;
Res{2}=SSS;
toc

save('Res.mat','Res')
