function [J1,J2]=fitfcn_val(w,error_idx,alpha1,alpha2,D,tr_idx,vl_idx,Exp,x0)

TNF_dose=[.5 1 2];


t_sam=0:1:14;
options = odeset('AbsTol',1e-6,'RelTol',1e-6);

W=reshape(w,length(error_idx),length(t_sam),length(TNF_dose));
   
W=W.*D;
tt=0:0.1:t_sam(end);

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
    y=deval(R1,tt);
    YY(i,:)=y(3,:);
    YYY(:,:,i)=y;
   [~,Yn]=ode15s(@(t,x) case_det(t,x,u),tt,x0,options);
   Yn=Yn';
   YYn(:,:,i)=Yn;
   yn(i,:)=Yn(3,:);
   
   [~,x]=ode15s(@(t,x) case_det_orig(t,x,u),tt,x0,options);
   Yo(:,:,i)=x';
end


  diff=(Y-Exp)./Exp;
  
  d1=sum(sum(diff(:,tr_idx).^2));
  J1=d1+(alpha2/2)*sum(reshape(W,length(error_idx)*length(t_sam)*length(TNF_dose),1).^2);
  
  if nargout>1
       J1=d1;
       J2=sum(sum(diff(:,vl_idx).^2));
      
  end
  
  for i=1:length(TNF_dose)
           figure(i)
           plot(t_sam,Exp(i,:),'bo','Markersize',15,'linewidth',3)
           hold on
           plot(tt,yn(i,:),'b--','linewidth',3)
      
           plot(tt, YY(i,:),'r-','linewidth',3)
           hold off
           xlabel('Time, hour')
           ylabel('NF\kappaB Activity')
           title(strcat('TNF\alpha=',num2str(TNF_dose(i))))
          % ylim([0.2 1.1])
           ax = gca; % current axes
           ax.FontSize = 30;
           ax.FontWeight='bold';
           x0=10;
           y0=-100;
           width=800;
           height=600;
           set(gcf,'units','points','position',[x0,y0,width,height])
           if i==3
              legend('Measurements','Before Estimation','After Estimation') 
           end
% % %       for j=1:4
% % %           figure(j)
% % %           
% % %           plot(tt,YYY(j,:,i),'b-')
% % %           hold on
% % %           plot(tt,YYn(j,:,i),'b--')
% % %           
% % %           plot(tt, Yo(j,:,i),'r-')
% % %           hold off
% % %           
% % %       end
%   end
  %hold off
end
