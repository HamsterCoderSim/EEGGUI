function [X,Y,domainFt,maLabeled] = genData(nSmps,mu,Sigma)

nSmpAll = sum(nSmps(:));
maLabeled = false(sum(nSmps(:)),1);maLabeled(1:sum(nSmps(1,:))) = 1;
X = [];
Y = nan(nSmpAll,1);
[nDomain,nCls] = size(mu);
domainFt = false(nSmpAll,nDomain);
nDim = length(mu{1});

for iDomain = 1:nDomain
	for iCls = 1:nCls
		mu1 = mu{iDomain,iCls};
		R1 = chol(Sigma{iDomain,iCls});
		z = repmat(mu1,nSmps(iDomain,iCls),1) + randn(nSmps(iDomain,iCls),nDim)*R1/5;
		domainFt(size(X,1)+1:size(X,1)+nSmps(iDomain,iCls),iDomain) = 1;
		Y(size(X,1)+1:size(X,1)+nSmps(iDomain,iCls)) = iCls;
		X = [X;z];
	end
end
X = X-repmat(mean(X,1),size(X,1),1);

end