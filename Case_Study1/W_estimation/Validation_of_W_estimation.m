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
%%
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
J=fitfcn_val(X_opt,error_idx,1,Alpha(Alpha_idx,2),DDx,Tr_IDX(:,1),[],Y_Exp,x0);
