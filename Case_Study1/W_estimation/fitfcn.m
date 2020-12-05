%% evaluate the objective function

function [J1,J2]=fitfcn(w,error_idx,alpha1,alpha2,D,tr_idx,vl_idx,Exp,x0)

TNF_dose=[.5 1 2];


t_sam=0:1:14;
options = odeset('AbsTol',1e-6,'RelTol',1e-6);

W=reshape(w,length(error_idx),length(t_sam),length(TNF_dose));
   
W=W.*D;


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
    
    Y(i,:)=y(3,:);

end


  diff=(Y-Exp)./Exp;
  
  d1=sum(sum(diff(:,tr_idx).^2));
  J1=d1+(alpha2/2)*sum(reshape(W,length(error_idx)*length(t_sam)*length(TNF_dose),1).^2);
  
  if nargout>1
       J1=d1;
       J2=sum(sum(diff(:,vl_idx).^2));
      
  end

end
