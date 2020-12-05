clc
clear

m3=0;
v3=0.01;

m4=0;
v4=0.01;
mu3= log((m3^2)/sqrt(v3^2+m3^2));
sigma3 = sqrt(log(v3^2/(m3^2)+1));
  


t=0:1:14;
obs=zeros(1,length(t));

mul=makedist('Lognormal','mu',m3,'sigma',v3);
mul_noise=random(mul,length(t),3);


add=makedist('normal','mu',m4,'sigma',v4);
add_noise=random(add,length(t),3);


%% finiding appropriate initial condition

x0=zeros(4,1);
options = odeset('AbsTol',1e-6,'RelTol',1e-6);
static=false;
count=1;
initial=x0;
u=0;
while ~static
    [t_1,y_1]=ode15s(@(t,x) case_det(t,x,u),[0 10],initial,options);

    dy=case_det(0,y_1(end,:),0);
    if sum(abs(dy)<1e-6)==length(dy)
        static=true;
    else
        initial=y_1(end,:);
        count=count+1;
    end
end

x0=y_1(end,:);

%%
TNF_dose=[.5 1 2];
for i=1:length(TNF_dose)
u=TNF_dose(i);

R1=ode15s(@(t,x) case_det(t,x,u),t,x0,options);

y=deval(R1,t);
% R2=ode15s(@(t,x) case_det(t,x,u),4:1:10,y(:,end),options);
% y=[y deval(R2,[5:1:10])];
Y(i,:)=y(3,:).*mul_noise(:,i)'+add_noise(:,i)';

%plot(t,Y(i,:),'s--',t,y(3,:),'o-')

end



%%
clr={'b','r','k'};

TNF_dose=[.5 1 2];
for i=1:length(TNF_dose)
u=TNF_dose(i);



[t1,y]=ode15s(@(t,x) case_det_wrong(t,x,u),[0 t(end)],x0,options);
markstr=strcat(clr{i},'o');
plot(t,Y(i,:),markstr,'Markersize',10,'MarkerFacecolor',clr{i})
hold on
linstr=strcat(clr{i},'-');
plot(t1,y(:,3),linstr,'linewidth',2)

end

hold off
xlabel('Time, hour')
ylabel('NF\kappaB Activity')
ylim([0.2 1.1])
ax = gca; % current axes
ax.FontSize = 20;
ax.FontWeight='bold';
X0=10;
y0=-100;
width=800;
height=600;
set(gcf,'units','points','position',[X0,y0,width,height])
lgd=legend('Measurement (TNF=0.5)','Prediction (TNF=0.5)','Measurement (TNF=1)','Prediction (TNF=1)','Measurement (TNF=2)','Prediction (TNF=2)');
lgd.NumColumns=3;
lgd.FontSize=15;
