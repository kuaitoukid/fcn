clear; close all; clc
for imid = 1 : 1000
    if imid ~= 59
        continue
    end
    im = imread(['./', num2str(imid), '.jpg']);
    fp = fopen(['./', num2str(imid), 'ori.txt'], 'r'); %, 'n', 'utf8');
    bboxes = [];
    line_id = 0;
    while 1
        strl = fgetl(fp);
        if strl == -1
            break
        end
        if int32(strl) == 0
            continue
        end
        line_id = line_id + 1;
        coords_strs = regexp(strl, ',', 'split');
        bbox = zeros(1, 8);
        for cid = 1 : 8
            coords_str = coords_strs{cid};
            if line_id == 1 && cid == 1
                coords_str = coords_str(3 : end);
            end
            current_num = 0;
            for chid = 1 : length(coords_str)
                if int32(coords_str(chid)) ~= 0
                    current_num = current_num * 10 + str2double(coords_str(chid));
                end
            end
            
            bbox(cid) = current_num;
        end
        bboxes = [bboxes; bbox];
        while 1
            txt = fgetl(fp);
            if int32(txt) == 0
                continue
            end
            break
        end
    end
    fclose(fp);
    fpre = fopen(['./gt_', num2str(imid), '.txt'], 'w');
    for bid = 1 : size(bboxes, 1)
        for cid = 1 : size(bboxes, 2)
            fprintf(fpre, '%d,', bboxes(bid, cid));
        end
        fprintf(fpre, '%s\n', 'kk');
        im = bitmapplot(bboxes(bid, [2 4]), bboxes(bid, [1 3]), im, ...
            struct('LineWidth', 3, 'Color', [1 0 0 1]));
        im = bitmapplot(bboxes(bid, [4 6]), bboxes(bid, [3 5]), im, ...
            struct('LineWidth', 3, 'Color', [0 1 0 1]));
        im = bitmapplot(bboxes(bid, [6 8]), bboxes(bid, [5 7]), im, ...
            struct('LineWidth', 3, 'Color', [0 0 1 1]));
        im = bitmapplot(bboxes(bid, [8 2]), bboxes(bid, [7 1]), im, ...
            struct('LineWidth', 3, 'Color', [1 1 0 1]));
    end
    fclose(fpre);
    imwrite(im, ['gt_', num2str(imid), '.jpg'], 'png');
end
    
