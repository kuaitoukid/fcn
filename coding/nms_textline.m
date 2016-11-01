function pick = nms_textline(boxes, overlap)
% top = nms(boxes, overlap)
% Non-maximum suppression. (FAST VERSION)
% Greedily select high-scoring detections and skip detections
% that are significantly covered by a previously selected
% detection.
%
% NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
% but an inner loop has been eliminated to significantly speed it
% up in the case of a large number of boxes

% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
%
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm


if isempty(boxes)
    pick = [];
    return;
end

x1 = boxes(:, 2);
y1 = boxes(:, 3);
x2 = boxes(:, 4); % + boxes(:, 1) - 1;
y2 = boxes(:, 5); % + boxes(:, 2) - 1;
if size(boxes, 2) >=5
    score = boxes(:, 1);
else
    score = ones(size(boxes, 1), 1);
end

area = (x2 - x1 + 1) .* (y2 - y1 + 1);
[~, I] = sort(score);

pick = score * 0;
counter = 1;
while ~isempty(I)
    last = length(I);
    i = I(last);
    pick(counter) = i;
    counter = counter + 1;
    
    xx1 = max(x1(i), x1(I(1 : last - 1)));
    yy1 = max(y1(i), y1(I(1 : last - 1)));
    xx2 = min(x2(i), x2(I(1 : last - 1)));
    yy2 = min(y2(i), y2(I(1 : last - 1)));
    
    w = max(0, xx2 - xx1 + 1);
    h = max(0, yy2 - yy1 + 1);
    
    inter = w .* h;
    % o = inter ./ (area(i) + area(I(1 : last-1)) - inter);
    % o = inter ./ max(area(i), area(I(1:last-1)));
    o = inter ./ min(area(i), area(I(1 : last - 1)));
    max(o);
    I = I(o <= overlap);
end
pick = pick(1 : (counter - 1));

for pid = 1 : length(pick)
    xx1 = max(x1(pick(pid)), x1);
    yy1 = max(y1(pick(pid)), y1);
    xx2 = min(x2(pick(pid)), x2);
    yy2 = min(y2(pick(pid)), y2);
    
    w = max(0, xx2 - xx1 + 1);
    h = max(0, yy2 - yy1 + 1);
    
    inter = w .* h;
    % o = inter ./ (area(i) + area(I(1 : last-1)) - inter);
    % o = inter ./ max(area(i), area(I(1:last-1)));
    o = inter ./ min(area(pick(pid)), area);
    tmp_score = score .* (o > overlap);
    [~, maxp] = max(tmp_score);
    pick(pid) = maxp;
end





