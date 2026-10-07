import cv2
import numpy as np

def get_length(x1, y1, x2, y2,vtx_list,frame_width):#根据直线像素坐标获取点云的实际长度
    '''
    函数功能为根据图像中直线两端点的像素坐标获取空间对应直线的实际长度
    x1, y1, x2, y2分别为直线两端点的横纵坐标
    vtx_list为点云数据的上限（该参数无需同学修改或设置，默认即可）
    frame_width为图像的一行有的点云数据数量（该参数无需同学修改或设置，默认即可）
    '''
    point1_id = int(y1*frame_width+x1)
    point2_id = int(y2*frame_width+x2)
    length = 0.0
    x1,y1,z1 = 0.0,0.0,0.0
    x2,y2,z2 = 0.0,0.0,0.0
    if point1_id < len(vtx_list) and point2_id < len(vtx_list):#没有超出范围
        x1,y1,z1 = float(vtx_list[point1_id][0]),float(vtx_list[point1_id][1]),float(vtx_list[point1_id][2])
        x2,y2,z2 = float(vtx_list[point2_id][0]),float(vtx_list[point2_id][1]),float(vtx_list[point2_id][2])
        
        if abs(x1-0) == 0 and abs(y1-0) == 0 and abs(z1-0) == 0: #该点的点云不为空
            return length
        elif abs(x2-0) == 0 and abs(y2-0)==0 and abs(z2-0)==0: 
            return length
        else:
            length = np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))
    return length
def check_line(img, length_threshold, vtx_list, frame_width):
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Canny边缘检测
    edges = cv2.Canny(blurred,50,100)
    
    # 使用概率霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                           minLineLength=20, maxLineGap=20)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 强制转换为int类型
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 计算实际长度
            length = get_length(x1, y1, x2, y2, vtx_list, frame_width)
            length *= 100.0  # 转换为厘米
            if length <= 0.0:
                continue  # 跳过无效线段
                
            # 绘制检测到的直线
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 显示长度
            text = f"{length:.1f} cm"
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # 绘制文本背景
            cv2.rectangle(img, 
                        (mid_x - text_w//2, mid_y - text_h - baseline), 
                        (mid_x + text_w//2, mid_y), 
                        (255, 255, 255), -1)
            
            # 绘制文本
            cv2.putText(img, text, 
                    (mid_x - text_w//2, mid_y - baseline), 
                    font, font_scale, (0, 0, 0), thickness)
    
    return img