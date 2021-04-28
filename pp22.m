function [y] = MLPclassificationPredict(w,X,nHidden,nLabels)
[nInstances,nVars] = size(X);
nHiddenLayers = length(nHidden);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:nHiddenLayers
    hiddenWeights{h-1} = reshape( w(offset+1:...
        offset + (nHidden(h-1)+1)*nHidden(h) ), ...
        nHidden(h-1)+1, nHidden(h));
    offset = offset+(nHidden(h-1)+1)*nHidden(h);
end
outputWeights = w(offset+1:offset+(nHidden(end)+1)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end)+1,nLabels);

ip = cell(1, nHiddenLayers);
fp = cell(1, nHiddenLayers);

y = zeros(nInstances, nLabels);
% Compute Output
for i = 1:nInstances
    ip{1} = [1, X(i,:)*inputWeights];
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        ip{h} = [1, fp{h-1}*hiddenWeights{h-1}];
        fp{h} = tanh(ip{h});
    end
    y(i,:) = fp{end}*outputWeights;
end
[~,y] = max(y,[],2);

end
