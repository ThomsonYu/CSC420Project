# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:16:56 2018

@author: bill
"""
import cv2
import matplotlib.pyplot as plt

tracker = cv2.TrackerCSRT_create()
video = cv2.VideoCapture("clip_id_easy_Medium.mp4")
template = cv2.imread("./Template/drumstick10.png")

w, h = template.shape[:2]

def tracking():
    success, frame = video.read()
    
    #template match the first frame
    #drumstick_loc = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED)
    
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(drumstick_loc)
    
    #top_left = max_loc
    bbox = cv2.selectROI(frame, False)
    tracker.init(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), bbox)
    #prev_loc = top_left
    
    display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #bot_right = (top_left[0] + h, top_left[1] + w)
    
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))    
    cv2.rectangle(display, p1, p2, (0, 255, 0), 2)
    
    plt.imshow(display)
    plt.show()
    
    while True:
        success, frame = video.read()
        if not success:
            break
        
        ok, loc = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        if ok:
            loc = tuple(map(int, loc))
            cv2.rectangle(frame, loc[:2], (loc[0] + h, loc[1] + w), (0, 255, 0), 2)
            
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.show()
            '''
            if loc != prev_loc:
                print(prev_loc[:2], loc[:2])
            else:
                print("failed")
            prev_loc = loc
            '''
        else:
            print("failed")

def capture_first_frame():
    success, frame = video.read()
    cv2.imwrite('frame.jpg', frame)

if __name__ == '__main__':
    #capture_first_frame()
    tracking()
