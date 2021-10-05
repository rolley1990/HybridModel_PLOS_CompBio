%% create Figure 6

clc
clear

load('Res_ANN_opt.mat')

for i=1:11
    for j=1:10
        temp=mean(squeeze(Res(:,:,i,j)));
       AIC_tr(i,j)=temp(1);
       AIC_over(i,j)=temp(2);
       AIC_Exp(i,j)=temp(3);
       AIC_Exp_ov(i,j)=temp(4);
    end
end
[X1,X2]=meshgrid(0:10,1:10);

XX2=X2';
XX1=X1';
scatter3(XX2(:),XX1(:),AIC_over(:),150,AIC_over(:),'LineWidth',2)
hold on
scatter3(8,4,AIC_over(5,8),150,AIC_over(5,8),'filled','LineWidth',2)
hold off

xlabel('Number of neurons in the 1st layer')
ylabel('Number of neurons in the 2nd layer')
zlabel('Average AIC_c')
           
          % ylim([0.2 1.1])
           ax = gca; % current axes
           ax.FontSize = 15;
           ax.FontWeight='bold';
           x0=10;
           y0=-100;
           width=800;
           height=700;
 set(gcf,'units','points','position',[x0,y0,width,height])
h = rotate3d;
set(h, 'ActionPreCallback', 'set(gcf,''windowbuttonmotionfcn'',@align_axislabel)')
set(h, 'ActionPostCallback', 'set(gcf,''windowbuttonmotionfcn'','''')')
set(gcf, 'ResizeFcn', @align_axislabel)
align_axislabel([], gca)
axislabel_translation_slider;
