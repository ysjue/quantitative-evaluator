#!/usr/bin/env python
#%%
from cv2 import CV_16UC1, EVENT_MOUSEMOVE
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from utils import get_largest_one_component, get_ND_bounding_box

draw_mode = False # true if mouse is pressed
global class_id
class_id = 1
# mouse callback function

def draw_circle(event,x,y,flags,param):
    global draw_mode, img, mask, np_img

    kernal_size = 6 if class_id == 1 else 3
    if event== cv.EVENT_MOUSEMOVE and draw_mode:
        for i in range(2 * kernal_size):
            for j in range(2 * kernal_size):
                x_i = max(min(x - kernal_size + j, shape[1] -1 ), 0)
                y_i = max(min(y - kernal_size + i, shape[0] -1 ), 0)
                mask[y_i,x_i] = class_id
                scalar = [0]*3
                scalar[-class_id] = 255 
                img[y_i,x_i] = scalar
    elif event == cv.EVENT_LBUTTONDOWN:
        draw_mode = not draw_mode
        


root = '/home/sean/laser_ws/src/cranial_test_10_07'
files = os.listdir(root)
reannotate_list = [33, 39, 62, 23, 60,56,43,48]
reannotate_list = []
annotaion_cases = [f.split('annotation')[0] for f in files if 'annotation' in f]
for file in files:
    # if '7_' in file or '5_' in file:
    #     continue
    file_dir = os.path.join(root, file)
    if 'color.jpg' not in file:
        # if 'annotation.jpg' in file:
        #     anno_img = cv.imread(file_dir)
        #     anno_img = np.array(anno_img[:,:,0], np.float32)
        #     print(file)
        #     plt.imshow(anno_img)
        #     plt.show()
        continue
        # if 'depth.png' in file or 'annotation.jpg' in file:
        #     continue
    # continue
    # if file.split('color')[0] in annotaion_cases and \
    #     file.split('color')[0] not in reannotate_list:
    #     continue
    img = cv.imread(file_dir)
    origin_img = img.copy()
    shape = img.shape[:-1]
    mask = np.zeros(img.shape[:-1], np.uint8)
    cv.namedWindow(file)
   
    cv.setMouseCallback(file,draw_circle)
    while(1):
        
        cv.imshow(file,img)
        
        k = cv.waitKey(1) & 0xFF
        # if k!=255:
        #     print(k)
        if k < 52 and k> 48:
            class_id = k - 48
            print('Annotate the label of class {0:}'.format(class_id))
    
        elif k == 27: 
            cv.destroyAllWindows()
            draw_mode = False
            break
        
        elif k == 100: # Press key d
            img = origin_img.copy()
            mask = np.zeros(img.shape[:-1], np.uint8)
            draw_mode = False
            print('delete current annotaion')
        elif k == 119: # Press 'w', denoting write
            
            write_path = os.path.join(root, file.replace('_color.jpg', '_annotation.jpg'))
            cv.imwrite(write_path, mask)
            print(file + ' has been saved to '+write_path)
            draw_mode = False
            cv.destroyAllWindows()
            break
        
        elif k == 115:  # Press 's', denoting segmentation
            draw_mode =False
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            
            pred = np.zeros(img.shape,np.uint8)
            seg = cv.GC_PR_BGD * np.ones_like(mask)

            # Using bounding box to reduce the image size
            box_min,box_max = get_ND_bounding_box(mask == 2, margin = 3)
            assert len(box_min) == 2 and len(box_max) == 2
            mask_bbox = np.zeros_like(mask)
            mask_bbox[box_min[0]:box_max[0]+1,box_min[1]:box_max[1]+1] = 1
            
            seg[mask == 2] = 0
            seg[mask == 1] = 1
            
            seg_bbox = seg[box_min[0]:box_max[0]+1,box_min[1]:box_max[1]+1]
            cv.grabCut(origin_img[box_min[0]:box_max[0]+1,box_min[1]:box_max[1]+1,:], seg_bbox, None, bgdModel,fgdModel,9)
            
            seg[box_min[0]:box_max[0]+1,box_min[1]:box_max[1]+1] = seg_bbox
            seg_mask = np.where((seg==2)|(seg==0),0,1).astype('uint8')
            seg_mask = get_largest_one_component(seg_mask)
            
            plt.imshow(seg_mask)
            plt.show()
            mask[seg_mask] = 1
            mask[seg_mask == False] = 2
            alpha = 0.6
            beta = (1.0 - alpha)
            img = origin_img.copy()
            pred[seg_mask>0] = [0,0,255]

            img = cv.addWeighted(img, alpha, pred, beta, 0.0)
            

            # cv.destroyAllWindows()
            # break
    
cv.destroyAllWindows()
    