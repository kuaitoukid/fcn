load('c4test-bboxes.mat')
clc
% fp = fopen('c4prop.txt', 'r');
imid = 1;
while 1
    if imid > length(bboxes)
        break
    end
    %% prepare
%     if imid ~= 20 % 11 20
%         imid = imid + 1;
%         continue
%     end
    %     imname = fgetl(fp);
    %     if imname == -1
    %         break
    %     end
    %     strs = regexp(imname, ' ', 'split');
    %     imname = strs{1};
    imname = ['img_', num2str(imid), '.jpg'];
    im = imread(['c4-test-color/', imname]) * 1;
    [imh, imw, ~] = size(im);
    if size(im, 3) == 1
        im = repmat(im, [1, 1, 3]);
    end
    bbox = bboxes{imid};
    bbox = bbox(2 : end, :);
    %% build heatmap
    heatmap = 0 * ones(round(imh / 4), round(imw / 4));
    for bid = 1 : size(bbox, 1)
        for hh = bbox(bid, 7) :  min(bbox(bid, 7) + floor(1 / bbox(bid, 8)), size(heatmap, 1))
            for ww = bbox(bid, 6) : min(bbox(bid, 6) + floor(1 / bbox(bid, 8)), size(heatmap, 2))
                if heatmap(hh, ww) < bbox(bid, 1)
                    heatmap(hh, ww) = bbox(bid, 1);
                end
            end
        end
    end
%     imagesc(heatmap)
    conf_regions = bwlabeln(heatmap > 0.5);
    region_num = max(conf_regions(:));
    heatmap_indexes = bbox(:, 7) + (bbox(:, 6) - 1) * size(heatmap, 1);
    %% extract text lines for each region
    proto_textlines = {}; % finally we use denseboxes to refine the proto_textlines in case of imcomplete box size for short words like 'a'
    for rid = 1 : region_num
        cc = double(conf_regions == rid);
        [xgrid, ygrid] = meshgrid(1 : size(heatmap, 2), 1 : size(heatmap, 1));
        xgrid = xgrid .* cc;
        xgrid = xgrid(xgrid > 0);
        ygrid = ygrid .* cc;
        ygrid = ygrid(ygrid > 0);
        indexes_in_region = ygrid + (xgrid - 1) * size(heatmap, 1);
        bbox_in_region = [];
        for id_in_region = 1 : length(indexes_in_region)
            bbox_in_region = [bbox_in_region; ...
                bbox(heatmap_indexes == indexes_in_region(id_in_region), :)];
        end
        
        bbox_in_region = bbox_in_region(:, [1 2 3 4 5]);
%         box_height = (bbox_in_region(:, 5) - bbox_in_region(:, 3) + 1);
%         bbox_in_region(:, 2) = bbox_in_region(:, 2) - box_height / 2;
%         bbox_in_region(:, 4) = bbox_in_region(:, 4) + box_height / 2;
        
        pick = nms_textline(bbox_in_region, 0.5);
        bbox_in_region = bbox_in_region(pick, :);
        
        bbox_in_region(:, 1) = -bbox_in_region(:, 1);
        bbox_in_region = sortrows(bbox_in_region);
        bbox_in_region = ceil(bbox_in_region(:, 2 : 5));

        bbox_in_region(:, 1) = max(bbox_in_region(:, 1), 1);
        bbox_in_region(:, 2) = max(bbox_in_region(:, 2), 1);
        bbox_in_region(:, 3) = min(bbox_in_region(:, 3), imw);
        bbox_in_region(:, 4) = min(bbox_in_region(:, 4), imh);

        left = bbox_in_region(:, 1);
        top = bbox_in_region(:, 2);
        right = bbox_in_region(:, 3);
        bottom = bbox_in_region(:, 4);
        area = (right - left + 1) .* (bottom - top + 1);

        % set the text line directions
        bbox_angle = 1000 * ones(size(bbox_in_region, 1), 1);
        
        for bid = 1 : size(bbox_in_region, 1)
            if size(bbox_in_region, 1) == 1
                proto_textlines = [proto_textlines; {rid}, {[0, mean(bbox_in_region([1, 3])), mean(bbox_in_region([2, 4]))]}, {bbox_in_region}];
            end
            % find overlapped boxes
            left = max(bbox_in_region(:, 1), bbox_in_region(bid, 1));
            top = max(bbox_in_region(:, 2), bbox_in_region(bid, 2));
            right = min(bbox_in_region(:, 3), bbox_in_region(bid, 3));
            bottom = min(bbox_in_region(:, 4), bbox_in_region(bid, 4));
            wid = max(0, right - left + 1);
            hei = max(0, bottom - top + 1);
            inter = wid .* hei;
            inter = inter ./ (area + area(bid) - inter);    
            inter(bid) = 0;
            [maxv, maxp] = max(inter);
            pair_bbox = bbox_in_region(maxp, :);
            centerx = mean(bbox_in_region(bid, [1, 3]));
            centery = mean(bbox_in_region(bid, [2, 4]));

            centerx_pair = mean(pair_bbox([1, 3]));
            centery_pair = mean(pair_bbox([2, 4]));

            im = bitmapplot([centery centery_pair], [centerx centerx_pair], im, struct('LineWidth', 2, 'Color', [1 0 1 1]));
            bbox_angle(bid) = atand( (centery - centery_pair) / (centerx - centerx_pair) );
        end
        bbox_angle = bbox_angle(bbox_angle < 1000);
        if isempty(bbox_angle)
            continue;
        end
        region_angle = median(bbox_angle);
        %%
        lineid = zeros(size(bbox_in_region, 1), 1); % record whether current box has been crossed
        point_online_all = [];
        for bid = 1 : size(bbox_in_region, 1)
            if lineid(bid) > 0
                continue
            end
            lineid(bid) = max(lineid) + 1;
            point_online = [mean(bbox_in_region(bid, [1, 3])), mean(bbox_in_region(bid, [2, 4]))];
            point_online_all = [point_online_all; point_online];
            for bid2 = 1 : size(bbox_in_region, 1)
                if bid == bid2 || lineid(bid2) > 0
                    continue
                end
                if cross_line(region_angle, point_online, bbox_in_region(bid2, :))
                    lineid(bid2) = lineid(bid);
                end
            end
        end
        for lid = 1 : max(lineid)
            bboxes_inline = bbox_in_region(lineid == lid, :);
            proto_textlines = [proto_textlines; {rid}, {[region_angle, point_online_all(lid, :)]}, {bboxes_inline}];
        end
    end
    
    %% refine textline
    complete_textlines = [];
    need_refine = true;
    if need_refine
        for rid = 1 : region_num
            cc = double(conf_regions == rid);
            [xgrid, ygrid] = meshgrid(1 : size(heatmap, 2), 1 : size(heatmap, 1));
            xgrid = xgrid .* cc;
            xgrid = xgrid(xgrid > 0);
            ygrid = ygrid .* cc;
            ygrid = ygrid(ygrid > 0);
            indexes_in_region = ygrid + (xgrid - 1) * size(heatmap, 1);
            bbox_in_region = [];
            for id_in_region = 1 : length(indexes_in_region)
                bbox_in_region = [bbox_in_region; ...
                    bbox(heatmap_indexes == indexes_in_region(id_in_region), :)];
            end
            bbox_in_region = bbox_in_region(:, [1 2 3 4 5]);
            pick = nms_textline(bbox_in_region, 0.9);
            bbox_in_region = bbox_in_region(pick, :);
            bbox_in_region = bbox_in_region(:, 2 : 5);
%             bbox_used = zeros(size(bbox_in_region, 1), 1);
            for lid = 1 : size(proto_textlines, 1)
                region_id = proto_textlines(lid, 1);
                region_id = region_id{:};
                if region_id ~= rid
                    continue
                end
                proto_line = proto_textlines(lid, 3);
                proto_line = proto_line{:};
                region_angle = proto_textlines(lid, 2);
                region_angle = region_angle{:};
                point_online = region_angle(2 : 3);
                region_angle = region_angle(1);
                proposal_boxes = [];
                for bid = 1 : size(bbox_in_region, 1)
                    if cross_line(region_angle, point_online, bbox_in_region(bid, :))
                        proposal_boxes = [proposal_boxes; bbox_in_region(bid, :)];
                    end
                end
                if isempty(proposal_boxes) % should not be empty!
                    continue
                end
                if abs(region_angle) > 45 % verticle
                    [~, minp] = min(proposal_boxes(:, 2));
                    [~, maxp] = max(proposal_boxes(:, 4));
                    topleft = proposal_boxes(minp, 1);
                    lefttop = proposal_boxes(minp, 2);
                    topright = proposal_boxes(minp, 3);
                    righttop = proposal_boxes(minp, 2);

                    bottomright = proposal_boxes(maxp, 3);
                    rightbottom = proposal_boxes(maxp, 4);
                    bottomleft = proposal_boxes(maxp, 1);
                    leftbottom = proposal_boxes(maxp, 4);
                else % horizon
                    [~, minp] = min(proposal_boxes(:, 1));
                    [~, maxp] = max(proposal_boxes(:, 3));
                    topleft = proposal_boxes(minp, 1);
                    lefttop = proposal_boxes(minp, 2);
                    topright = proposal_boxes(maxp, 3);
                    righttop = proposal_boxes(maxp, 2);

                    bottomright = proposal_boxes(maxp, 3);
                    rightbottom = proposal_boxes(maxp, 4);
                    bottomleft = proposal_boxes(minp, 1);
                    leftbottom = proposal_boxes(minp, 4);
                end
                complete_textlines = [complete_textlines; topleft, lefttop, topright, righttop, bottomright, rightbottom, bottomleft, leftbottom];
                im = bitmapplot(complete_textlines(end, 2 : 2 : end), complete_textlines(end, 1 : 2 : end), im, struct('LineWidth', 1, 'Color', [1 0 0 1]));
            end
            
        end
    else
        for lid = 1 : size(proto_textlines, 1)
            proto_line = proto_textlines(lid, 3);
            proto_line = proto_line{:};
            region_angle = proto_textlines(lid, 2);
            region_angle = region_angle{:};
            region_angle = region_angle(1);
            if abs(region_angle) > 45 % verticle
                [~, minp] = min(proto_line(:, 2));
                [~, maxp] = max(proto_line(:, 4));
                topleft = proto_line(minp, 1);
                lefttop = proto_line(minp, 2);
                topright = proto_line(minp, 3);
                righttop = proto_line(minp, 2);
                
                bottomright = proto_line(maxp, 3);
                rightbottom = proto_line(maxp, 4);
                bottomleft = proto_line(maxp, 1);
                leftbottom = proto_line(maxp, 4);
            else % horizon
                [~, minp] = min(proto_line(:, 1));
                [~, maxp] = max(proto_line(:, 3));
                topleft = proto_line(minp, 1);
                lefttop = proto_line(minp, 2);
                topright = proto_line(maxp, 3);
                righttop = proto_line(maxp, 2);
                
                bottomright = proto_line(maxp, 3);
                rightbottom = proto_line(maxp, 4);
                bottomleft = proto_line(minp, 1);
                leftbottom = proto_line(minp, 4);
            end
            complete_textlines = [complete_textlines; topleft, lefttop, topright, righttop, bottomright, rightbottom, bottomleft, leftbottom];
            im = bitmapplot(complete_textlines(end, 2 : 2 : end), complete_textlines(end, 1 : 2 : end), im, struct('LineWidth', 1, 'Color', [0 1 1 1]));
        end
    end
    
%     imshow(im)
    imwrite(im, [imname, '-gt.png'], 'png');
    imid = imid + 1;
    %         break
end

% fclose(fp);


















