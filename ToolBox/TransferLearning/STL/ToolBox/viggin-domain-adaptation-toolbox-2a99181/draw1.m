function draw1(X,Y,domainFt,axlb,name,acc)

% figure
hold on, %axis equal
co = {'r.','ro';'b+','bs';'g*','gd'};
nDomain = size(domainFt,2);
nCls = max(Y);
for iDomain = 1:nDomain
	for iCls = 1:nCls
		ma = domainFt(:,iDomain)&Y==iCls;
		plot(X(ma,1),X(ma,2),co{iDomain,iCls})
	end
end
xlabel(axlb{1})
ylabel(axlb{2})
title(sprintf('%s, %.2f%%',name,acc*100))

end