function delta=case_det_addW(t,x,u,PP,error_idx)

delta = case_det1(t,x,u);
e=zeros(length(error_idx),1);
for j=1:length(error_idx)
    
    pp=PP{j};
    e(j)=ppval(pp,t);
    
end

delta(error_idx)=delta(error_idx)+e;


end
function J=case_det1(t,x,u)
J=zeros(4,1);
a1=0.6;
a2=0.2;
a3=0.2;
a4=0.5;

b1=0.4;
b2=0.7;
b3=0.3;
b4=0.5;
b5=0.4;

J(1)=-x(1)+0.5*((b4^2/(b4^2+x(3)^2))*(u^2/(a1^2+u^2))+(x(2)^2/(a3^2+x(2)^2)) );
J(2)=-x(2)+(x(1)^2/(a2^2+x(1)^2))*(b3^2/(b3^2+x(3)^2));
%     J(3)=-x(3)+(b2^2/(b2^2+x(2)^2))*(b5^2/(b5^2+x(4)^2));
J(3)=-x(3)+(b5^2/(b5^2+x(4)^2));
J(4)=-x(4)+0.5*((b1^2/(b1^2+u^2))+(x(3)^2/(a4^2+x(3)^2)));


end