function NewDots = rearrange_vertex_seq(dots)
Len_12 = ((dots(1)-dots(3))^2+(dots(2)-dots(4))^2)^0.5;
Len_23 = ((dots(3)-dots(5))^2+(dots(4)-dots(6))^2)^0.5;
Len_34 = ((dots(5)-dots(7))^2+(dots(6)-dots(8))^2)^0.5;
Len_14 = ((dots(1)-dots(7))^2+(dots(2)-dots(8))^2)^0.5;
if mean([Len_12,Len_34]) >= mean([Len_14,Len_23])
    x1 = (dots(1)+dots(7))*0.5;
    x2 = (dots(3)+dots(5))*0.5;
    y1 = (dots(2)+dots(8))*0.5;
    y2 = (dots(4)+dots(6))*0.5;
else
    x1 = (dots(1)+dots(3))*0.5;
    x2 = (dots(5)+dots(7))*0.5;
    y1 = (dots(2)+dots(4))*0.5;
    y2 = (dots(6)+dots(8))*0.5;
end
theta= atan2(abs(y1-y2),abs(x1-x2));
if theta > pi/4
    if mean([Len_12,Len_34]) >= mean([Len_14,Len_23])
        if dots(2)<=dots(4)
            if dots(1)<=dots(7)
                NewDots(1)=dots(1);
                NewDots(2)=dots(2);
                NewDots(3)=dots(7);
                NewDots(4)=dots(8);
                NewDots(5)=dots(5);
                NewDots(6)=dots(6);
                NewDots(7)=dots(3);
                NewDots(8)=dots(4);
            else
                NewDots(1)=dots(7);
                NewDots(2)=dots(8);
                NewDots(3)=dots(1);
                NewDots(4)=dots(2);
                NewDots(5)=dots(3);
                NewDots(6)=dots(4);
                NewDots(7)=dots(5);
                NewDots(8)=dots(6);
            end
        else
            if dots(1)<=dots(7)
                NewDots(1)=dots(3);
                NewDots(2)=dots(4);
                NewDots(3)=dots(5);
                NewDots(4)=dots(6);
                NewDots(5)=dots(7);
                NewDots(6)=dots(8);
                NewDots(7)=dots(1);
                NewDots(8)=dots(2);
            else
                NewDots(1)=dots(5);
                NewDots(2)=dots(6);
                NewDots(3)=dots(3);
                NewDots(4)=dots(4);
                NewDots(5)=dots(1);
                NewDots(6)=dots(2);
                NewDots(7)=dots(7);
                NewDots(8)=dots(8);
            end
        end
    else
        if dots(4)<=dots(6)
            if dots(1)<=dots(3)
                NewDots(1)=dots(1);
                NewDots(2)=dots(2);
                NewDots(3)=dots(3);
                NewDots(4)=dots(4);
                NewDots(5)=dots(5);
                NewDots(6)=dots(6);
                NewDots(7)=dots(7);
                NewDots(8)=dots(8);
            else
                NewDots(1)=dots(3);
                NewDots(2)=dots(4);
                NewDots(3)=dots(1);
                NewDots(4)=dots(2);
                NewDots(5)=dots(7);
                NewDots(6)=dots(8);
                NewDots(7)=dots(5);
                NewDots(8)=dots(6);
            end
        else
            if dots(1)<=dots(3)
                NewDots(1)=dots(7);
                NewDots(2)=dots(8);
                NewDots(3)=dots(5);
                NewDots(4)=dots(6);
                NewDots(5)=dots(3);
                NewDots(6)=dots(4);
                NewDots(7)=dots(1);
                NewDots(8)=dots(2);
            else
                NewDots(1)=dots(5);
                NewDots(2)=dots(6);
                NewDots(3)=dots(7);
                NewDots(4)=dots(8);
                NewDots(5)=dots(1);
                NewDots(6)=dots(2);
                NewDots(7)=dots(3);
                NewDots(8)=dots(4);
            end
        end
    end
else
    if mean([Len_12,Len_34]) >= mean([Len_14,Len_23])
        if dots(1)<=dots(3)
            if dots(2)<=dots(8)
                NewDots(1)=dots(1);
                NewDots(2)=dots(2);
                NewDots(3)=dots(3);
                NewDots(4)=dots(4);
                NewDots(5)=dots(5);
                NewDots(6)=dots(6);
                NewDots(7)=dots(7);
                NewDots(8)=dots(8);
            else
                NewDots(1)=dots(7);
                NewDots(2)=dots(8);
                NewDots(3)=dots(5);
                NewDots(4)=dots(6);
                NewDots(5)=dots(3);
                NewDots(6)=dots(4);
                NewDots(7)=dots(1);
                NewDots(8)=dots(2);
            end
        else
            if dots(4)<=dots(6)
                NewDots(1)=dots(3);
                NewDots(2)=dots(4);
                NewDots(3)=dots(1);
                NewDots(4)=dots(2);
                NewDots(5)=dots(7);
                NewDots(6)=dots(8);
                NewDots(7)=dots(5);
                NewDots(8)=dots(6);
            else
                NewDots(1)=dots(5);
                NewDots(2)=dots(6);
                NewDots(3)=dots(7);
                NewDots(4)=dots(8);
                NewDots(5)=dots(1);
                NewDots(6)=dots(2);
                NewDots(7)=dots(3);
                NewDots(8)=dots(4);
            end
        end
    else
        if dots(3)<=dots(5)
            if dots(2)<=dots(4)
                NewDots(1)=dots(1);
                NewDots(2)=dots(2);
                NewDots(3)=dots(7);
                NewDots(4)=dots(8);
                NewDots(5)=dots(5);
                NewDots(6)=dots(6);
                NewDots(7)=dots(3);
                NewDots(8)=dots(4);
            else
                NewDots(1)=dots(3);
                NewDots(2)=dots(4);
                NewDots(3)=dots(5);
                NewDots(4)=dots(6);
                NewDots(5)=dots(7);
                NewDots(6)=dots(8);
                NewDots(7)=dots(1);
                NewDots(8)=dots(2);
            end
        else
            if dots(2)<=dots(4)
                NewDots(1)=dots(7);
                NewDots(2)=dots(8);
                NewDots(3)=dots(1);
                NewDots(4)=dots(2);
                NewDots(5)=dots(3);
                NewDots(6)=dots(4);
                NewDots(7)=dots(5);
                NewDots(8)=dots(6);
            else
                NewDots(1)=dots(5);
                NewDots(2)=dots(6);
                NewDots(3)=dots(3);
                NewDots(4)=dots(4);
                NewDots(5)=dots(1);
                NewDots(6)=dots(2);
                NewDots(7)=dots(7);
                NewDots(8)=dots(8);
            end
        end
    end
end

