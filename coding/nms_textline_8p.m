function pick = nms_textline_8p(boxes, overlap)
if isempty(boxes)
    pick = [];
    return;
end
score = boxes(:, 1);
[~, I] = sort(score);

pick = score * 0;
counter = 1;

areas = zeros(size(boxes, 1), 1);


for bid = 1 : size(boxes, 1)
    areas(bid) = polyarea(boxes(bid, 2 : 2 : 9), boxes(bid, 3 : 2 : 9));
end

overlap_all = -1 * ones(size(boxes, 1)) + 2 * eye(size(boxes, 1));

while ~isempty(I)
    last = length(I);
    i = I(last);
    pick(counter) = i;
    counter = counter + 1;
    o = zeros(1, last - 1);
    for oid = 1 : last - 1
        o(oid) = quad_overlap(boxes(i, 2 : 9), boxes(I(oid), 2 : 9), areas(i), areas(I(oid)));
        overlap_all(i, I(oid)) = o(oid);
        overlap_all(I(oid), i) = o(oid);
    end
    I = I(o <= overlap);
end
pick = pick(1 : (counter - 1));

for bid1 = 1 : size(boxes, 1) - 1
    if isempty(find(pick == bid1, 1))
        continue
    end
    for bid2 = bid1 + 1 : size(boxes, 1)
        if overlap_all(bid1, bid2) >= 0
            continue
        end
        overlap_all(bid1, bid2) = quad_overlap(boxes(bid1, 2 : 9), boxes(bid2, 2 : 9), areas(bid1), areas(bid2));
        overlap_all(bid2, bid1) = overlap_all(bid1, bid2);
    end
end

for pid = 1 : length(pick)
    o = overlap_all(:, pick(pid));
    tmp_score = score .* (o > overlap);
    [~, maxp] = max(tmp_score);
    pick(pid) = maxp;
end





