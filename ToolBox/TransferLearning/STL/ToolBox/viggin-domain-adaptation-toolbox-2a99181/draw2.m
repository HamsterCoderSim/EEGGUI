function draw2(X,Y,domainFt,axlb,name,acc)
% diff time smp

hold on
co = jet(nnz(domainFt(:,1)~=0&Y==1)); mo = '.o';
nDomain = size(domainFt,2)/2;
nCls = max(Y);

% for legend
for iDomain = 1:nDomain
	for iSmp = [1 90]
		for iCls = 1:nCls
			id = find(domainFt(:,iDomain*2-1)~=0&Y==iCls);
			[~,ord] = sort(domainFt(id,2),'ascend');
			ordr = 1:length(id); ordr(ord) = ordr;
			plot(X(id(iSmp),1),X(id(iSmp),2),mo(iCls),'color',co(ordr(iSmp),:))
		end
	end
end

for iDomain = 1:nDomain
	for iCls = 1:nCls
		id = find(domainFt(:,iDomain*2-1)~=0&Y==iCls);
		[~,ord] = sort(domainFt(id,2),'ascend');
		ordr = 1:length(id); ordr(ord) = ordr;
		for iSmp = 1:length(id)
			plot(X(id(iSmp),1),X(id(iSmp),2),mo(iCls),'color',co(ordr(iSmp),:))
		end
	end
end
xlabel(axlb{1})
ylabel(axlb{2})
title(sprintf('%s, %.2f%%',name,acc*100))

end