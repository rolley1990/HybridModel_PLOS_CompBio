function [Y,Eo]=fitfcn(w,error_idx,alpha1,alpha2,D,tr_idx,vl_idx,x0,Exp)

TNF_dose=[.5 1 2];


t_sam=0:1:14;
options = odeset('AbsTol',1e-6,'RelTol',1e-6);

W=reshape(w,length(error_idx),length(t_sam),length(TNF_dose));
   
W=W.*D;
tt=0:0.5:t_sam(end);

for i=1:length(TNF_dose)
    
    E=squeeze(W(:,:,i));    
    PP=cell(length(error_idx),1);
    for j=1:length(error_idx)
        pp= interp1(t_sam,E(j,:),'linear','pp');
        PP{j}=pp;
    end
    
    u=TNF_dose(i);
    
    R1=ode15s(@(t,x) case_det_addW(t,x,u,PP,error_idx),t_sam,x0,options);
    
    y=deval(R1,t_sam);
    ym(i,:)=y(3,:);
%    y=deval(R1,tt);
    Y(:,:,i)=y;
  
    figure(i)
    plot(t_sam,Exp(i,:),'ro')
    hold on
    plot(tt,y(3,:))
    hold off
end

Eo=mean(mean(((ym-Exp)./Exp).^2));

   %hold off
end