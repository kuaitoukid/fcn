function crossed = cross_line(theta, point_online, bbox)
crossed = false;
% tand(theta) * x - y - (tand(theta) * linecenter(1) - linecenter(2))
v = [0 0 0 0];
x = bbox(1);
y = bbox(2);
v(1) = tand(theta) * x - y - (tand(theta) * point_online(1) - point_online(2));

x = bbox(3);
y = bbox(2);
v(2) = tand(theta) * x - y - (tand(theta) * point_online(1) - point_online(2));

x = bbox(3);
y = bbox(4);
v(3) = tand(theta) * x - y - (tand(theta) * point_online(1) - point_online(2));

x = bbox(1);
y = bbox(4);
v(4) = tand(theta) * x - y - (tand(theta) * point_online(1) - point_online(2));

if min(v) * max(v) <= 0
    crossed = true;
end