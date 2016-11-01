%% Two Quadrilaterals Intersection Area (Special Case)
function overlap = quad_overlap(v1, v2, area1, area2)
overlap = 0;
v1 = reshape(v1, [2, 4])';
v2 = reshape(v2, [2, 4])';
if nargin < 2
    v1 = [15.4300   14.7200;  938.7000   15.1300;  932.4300  205.8500;   18.3300  203.6600];
    v2 = [16.4900   16.2300;  937.2500   16.3500;  932.6900  205.3400;   19.1800  201.7600];
    v1 = [0 0; 2 0; 2 2; 0 2];
    v2 = [1 1; 3 4; 0.12 1.5; 0.12 1];
    v1 = [0 0; 2 0; 2 2; 0 2] + 1000;
    v2 = [1 1; 3 4; 0.12 1.5; 0.12 1];
elseif nargin < 4
    area1 = polyarea(v1(:, 1), v1(:, 2));
    area2 = polyarea(v2(:, 1), v2(:, 2));
end

xbox1 = [v1(:, 1); v1(1, 1)];
ybox1 = [v1(:, 2); v1(1, 2)];
xbox2 = [v2(:, 1); v2(1, 1)];
ybox2 = [v2(:, 2); v2(1, 2)];

[interx, intery] = polyxpoly(xbox1, ybox1, xbox2, ybox2, 'unique');
inter_pt = [interx, intery];
% hold on
% plot(xbox1, ybox1, '.-')
% plot(xbox2, ybox2, 'r.-')
% plot(interx, intery, 'go')

box1inbox2 = inpolygon(v1(:, 1), v1(:, 2), xbox2, ybox2);
inter_pt = [inter_pt; v1(box1inbox2, :)];
box2inbox1 = inpolygon(v2(:, 1), v2(:, 2), xbox1, ybox1);
inter_pt = [inter_pt; v2(box2inbox1, :)];

if size(inter_pt, 1) < 3
    return
end
removed_id = [];
for pid = 1 : size(inter_pt, 1) - 1
    for pid2 = pid + 1 : size(inter_pt, 1)
        if inter_pt(pid, 1) == inter_pt(pid2, 1) && inter_pt(pid, 2) == inter_pt(pid2, 2)
            removed_id = [removed_id, pid2];
        end
    end
end
inter_pt(removed_id, :) = [];
if size(inter_pt, 1) < 3
    return
end

vi = convhull(inter_pt(:, 1), inter_pt(:, 2));
inter_area = polyarea(inter_pt(vi, 1), inter_pt(vi, 2));
% area1 = polyarea(v1(:, 1), v1(:, 2));
% area2 = polyarea(v2(:, 1), v2(:, 2));
% overlap = inter_area / (area1 + area2 - inter_area);
overlap = inter_area / min(area1, area2);





