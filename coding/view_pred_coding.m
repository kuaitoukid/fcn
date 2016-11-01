load('c2test-bboxes-coding.mat')
clc
% fp = fopen('c4prop.txt', 'r');
imid = 1;
while 1
    if imid > length(bboxes)
        break
    end
    if imid < 39
        imid = imid + 1;
        continue
    end
%     imname = fgetl(fp);
%     if imname == -1
%         break
%     end
%     strs = regexp(imname, ' ', 'split');
%     imname = strs{1};
    imname = ['img_', num2str(imid), '.jpg'];
    im = imread(['c2-test-color/', imname]) * 1;
    [imh, imw, ~] = size(im);
    if size(im, 3) == 1
        im = repmat(im, [1, 1, 3]);
    end
    bbox = bboxes{imid};
    bbox = bbox(2 : end, :);
    bbox = bbox(bbox(:, 1) > 0.7, :);
    bbox = bbox(:, 1 : 9);
    tic
    pick = nms_textline_8p(bbox, 0.5);
    toc
    bbox = bbox(pick, :);
    bbox = ceil(bbox(:, 2 : 9));
    
    bbox(:, 1 : 2 : 8) = min(max(bbox(:, 1 : 2 : 8), 1), imw);
    bbox(:, 2 : 2 : 8) = min(max(bbox(:, 2 : 2 : 8), 1), imh);
    
    for bid = 1 : size(bbox, 1)
        centerx = round( mean(bbox(bid, 1 : 2 : 8)) );
        centery = round( mean(bbox(bid, 2 : 2 : 8)) );
        im(centery : centery + 0, centerx : centerx + 0, :) = 0;
        im(round(centery : centery + 0), round(centerx : centerx + 0), 1) = 255;
        
        im = bitmapplot(bbox(bid, [2 : 2 : 8, 2]), bbox(bid, [1 : 2 : 8, 1]), im, struct('LineWidth', 2, 'Color', [1 0 0 1]));
        
        %         if bbox(i, 1) < 1 || bbox(i, 2) < 1 || bbox(i, 3) > size(im, 2) || bbox(i, 4) > size(im, 1)
        %             continue
        %         end
        
        
%                 left = bbox(i, 1);
%                 right = bbox(i, 3);
%                 tmpw = right - left + 1;
%                 top = bbox(i, 2);
%                 bottom = bbox(i, 4);
%                 tmph = bottom - top + 1;
% %                 patch_size = tmph;
%                 if max(tmpw, tmph) < 5
%                     continue
%                 end
        %         if tmpw > tmph
        %             patch_size = tmpw;
        %             bottom = round(bottom - (tmpw - tmph) / 2);
        %             top = round(top + (tmpw - tmph) / 2);
        %         else
        %             left = round(left - (tmph - tmpw) / 2);
        %             right = round(right + (tmph - tmpw) / 2);
        %         end
%                 left = max(left, 1);
%                 top = max(top, 1);
%                 right = min(right, imw);
%                 bottom = min(bottom, imh);
%                 tmpim = im(top : bottom, left : right, :);
        %         if isempty(tmpim)
        %             continue
        %         end
%                 patch_size = 32;
%                 tmpim = imresize(tmpim, [patch_size, patch_size]);
%                 imwrite(tmpim, ['c2-test-blocks/', num2str(imid), '_', num2str(i), '.png'], 'png');
    end
%     figure
%     imshow(im)
    imwrite(im, [imname, '-gt.png'], 'png');
    imid = imid + 1;
%     break
end

% fclose(fp);


















