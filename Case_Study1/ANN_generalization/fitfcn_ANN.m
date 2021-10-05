function [J,G,J2]=fitfcn_ANN(HP,error_idx,alpha,tr_idx,vl_idx,x0,Exp,West)

TNF_dose=[.5 1 2];


t_sam=0:1:14;
options = odeset('AbsTol',1e-6,'RelTol',1e-6);



for i=1:length(TNF_dose)
    
    u=TNF_dose(i);
    
   R1=ode15s(@(t,x) case_det_ANN(t,x,u,HP,error_idx),t_sam,x0,options);
    y=deval(R1,t_sam);
    Y(i,:)=y(3,:);
    w=myNeuralNetworkFunction([y;t_sam],HP);
    wtemp(i,:)=((w-West(i,:))./West(i,:)).^2;
    
end

    diff=(Y-Exp)./Exp;
  
  d1=mean(mean(diff(:,tr_idx).^2));
  d2= mean(mean(wtemp(:,tr_idx).^2));

  
  J=d1+d2+(alpha/2)*sum(HP.^2);
  
  if nargout==2
     G=0;
      St=zeros(length(t_sam),length(HP),length(TNF_dose));
      dPdW=zeros(length(t_sam),length(HP),length(TNF_dose));
      g=zeros(length(HP),1);
      for i=1:length(TNF_dose)
          
          u=TNF_dose(i);
          
          R1=ode15s(@(t,x) case_det_ANN(t,x,u,HP,error_idx),t_sam,x0,options);
          y=deval(R1,t_sam);
          S0=zeros(4,1);
          for j=1:length(HP)
              [~,s]=ode15s(@(t,x) case_det_J(t,x,R1,HP,u,j),t_sam,S0,options);
              St(:,j,i)=s(:,3);
              for k=1:length(t_sam)
                  [~, dPdW(k,j,i)]=case_det_J(t_sam(k),s(k,:)',R1,HP,u,j);
              end
          end
          ytemp=-2*((Y(i,:)-Exp(i,:))./Exp(i,:)).^2;
          Ytemp=ytemp*St(:,:,i);
          Ytemp=Ytemp';
%           w=myNeuralNetworkFunction([y;t_sam],HP);
%           wtemp=-2*((w-West(i,:))./West(i,:)).^2;
          Wtemp=-2*wtemp(i,:)*dPdW(:,:,i);
          Wtemp=Wtemp';
          g=Ytemp+Wtemp;
          G=G+g';
      end
      
  end   
  if nargout==3
       diff=(Y-Exp)./Exp;
  
       J=mean(mean(diff.^2));
        J2= mean(mean(wtemp.^2));

      
  end
  
end