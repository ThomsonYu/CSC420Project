import cv2
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import os

video_link = "clip_id_easy_Medium.mp4"
hit_audio = "noise_chopped.wav"
tracker = cv2.TrackerCSRT_create()
video = cv2.VideoCapture(video_link)

def tracking(visualize=False):
    success, frame = video.read()
    init_frame = np.copy(frame)

    vh, vw = frame.shape[:2]

    '''
    #when selecting objects from a video clip the first time, copy down the location so no need to select it everytime    
    bbox = cv2.selectROI(frame, False)
    drum_box = cv2.selectROI(frame, False)
    person_box = cv2.selectROI(frame, False)

    print(bbox)
    print(drum_box)
    print(person_box)
    return
    '''
    
    '''
    #drumclip10
    bbox = (750, 325, 55, 58)
    drum_box = (718, 347, 197, 233)
    person_box =(497, 230, 100, 413)
    '''
    
    
    #drumclip11
    bbox = (859, 304, 47, 45)
    drum_box = (801, 125, 309, 454)
    person_box = (577, 7, 191, 640)
    
    
    h, w = bbox[2:]
    dh, dw = drum_box[2:]
    tracker.init(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), bbox)
    
    display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))    

    frames = []
    locs = []
    while True:
        success, frame = video.read()
        if not success:
            break
        
        ok, loc = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        #we are only including the frames where tracking is sucessful, because no point of analysis if tracking fails
        if ok:
            loc = tuple(map(int, loc))
            frames.append(frame)
            locs.append(loc[:2])
        else:
            print("failed")

    '''
    #Naive way of checking contact(i.e. check if drumstick collides with drum hitbox in 2d
    contact_index = contact_made_naive(drum_box, locs, (h, w), frames)
    process_frames(frames, locs, (h, w), visualize, drum_box=drum_box)
    '''
    
    '''
    #contact method 1
    contact_index, bar = contact_made(person_box[:2], drum_box[:2], locs, frames)
    #visualize tracking and threshold
    process_frames(frames, locs, (h, w), visualize, bar=bar)
    '''

    
    #contact method 2
    contact_index = contact_made_2(person_box[:2], drum_box[:2], locs, frames)

    if visualize:
        process_frames(frames, locs, (h, w), False)    #No threshold bar to visualize    
        
        visualize_contact(contact_index, frames)

        visualize_drumstick_path(init_frame, locs, (h, w))

    #writing the frames where tracking is successful
    codec = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter('output.mp4', codec, 20, (vw, vh))

    for frame in frames:
        video_out.write(frame)

    video_out.release()

    #write the audio based on the contact frames
    out_audio = create_audio(frames, contact_index, hit_audio)
    out_audio.export('output_audio.wav', format="wav")

    #now we have both the video and the audio, combine them
    os.system("ffmpeg -i output.mp4 -i output_audio.wav -y -vcodec copy -acodec copy output_vid_aud.avi")
    

'''
Different methods of checking if contact has been made
'''
def contact_made_naive(drum_box, locs, stick_size, frames):
    sh, sw = stick_size
    
    contact_index = []
    
    for i in range(len(locs)):
        loc = locs[i]
        #if the drum_stick lie entirely within the drum box,
        #i.e left_side_drum < left_side_stick < right_size_stick < right_side_drum
        collide = drum_box[0] < loc[0] < loc[0] + sw < drum_box[0] + drum_box[2] and drum_box[1] < loc[1] < loc[1] + sh < drum_box[1] + drum_box[3]
        if collide:
            contact_index.append(i)
    return contact_index

def contact_made(loc_person, loc_drum, all_loc_sticks, frames, threshold=0.7):
    #threshold value can be tuned. larger value means more strict restriction
    dx, dy = loc_drum
    px, py = loc_person

    min_x = min(all_loc_sticks)[0]
    max_x = max(all_loc_sticks)[0]

    diff = max_x - min_x
    
    if loc_person[0] < loc_drum[0]:
        print("right")
        #drum is on the right side, so return the stick locations that are on the right
        bar = min_x + (threshold) * diff
        contact_index = [i for i in range(len(frames)) if all_loc_sticks[i][0] > min_x + (threshold) * diff]
    else:
        print("left")
        #drum is on the left side, so return the stick locations that are on the left
        bar = min_x + (1 - threshold) * diff
        contact_index = [i for i in range(len(frames)) if all_loc_sticks[i][0] < min_x + (1 - threshold) * diff]

    return contact_index, bar

def contact_made_2(loc_person, loc_drum, all_loc_sticks, frames):

    dx, dy = loc_drum
    px, py = loc_person

    contact_index = []
    if loc_person[0] < loc_drum[0]:
        print("right")
        #detect change in direction from right to left -> hit
        for i in range(1, len(frames) - 1):
            if all_loc_sticks[i][0] > all_loc_sticks[i - 1][0] and all_loc_sticks[i][0] > all_loc_sticks[i + 1][0]:
                contact_index.append(i)

    else:
        print("left")
        #detect change in direction from left to right -> hit
        for i in range(1, len(frames) - 1):
            if all_loc_sticks[i][0] < all_loc_sticks[i - 1][0] and all_loc_sticks[i][0] < all_loc_sticks[i + 1][0]:
                contact_index.append(i)

    return contact_index
    

'''
A few visualization methods
'''
def viz_threshold(frame, min_x, max_x, bar):

    #draw threshold
    frame[:, int(bar): int(bar) + 5] = np.array([255, 0, 0])
    frame[:, min_x: min_x + 5] = np.array([255, 0, 0])
    frame[:, max_x: max_x + 5] = np.array([255, 0, 0])

def viz_drum_loc(frame, drum_box):
    top_left = (drum_box[:2])
    bot_right = (top_left[0] + drum_box[2], top_left[1] + drum_box[3])
    cv2.rectangle(frame, top_left, bot_right, (255, 0, 0), thickness=5)

def process_frames(frames, locs, stick_size, visualize, bar=None, drum_box=None):
    sh, sw = stick_size
    
    min_x = min(locs)[0]
    max_x = max(locs)[0]
    
    for i in range(len(frames)):    
        frame = frames[i]
        loc = locs[i]
        cv2.rectangle(frame, loc, (loc[0] + sw, loc[1] + sh), (0, 255, 0), 2)
        if visualize:
            if bar:
                viz_threshold(frame, min_x, max_x, bar)

            if drum_box:
                viz_drum_loc(frame, drum_box)

def visualize_contact(contact_index, frames):
    w, h = frames[0].shape[:2]
    for index in contact_index:
        frame = frames[index]
        cv2.rectangle(frame, (h // 2 - 150, 50), (h // 2 + 150, 200), (255, 255, 255), thickness = -1)
        cv2.putText(frame, "boop", (h // 2 - 150, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), thickness=5)

def visualize_drumstick_path(init_frame, all_loc_sticks, stick_size):
    h, w = stick_size

    for loc in all_loc_sticks:
        cv2.circle(init_frame, (loc[0] + w // 2, loc[1] + h // 2), 2, (255, 0, 0), thickness=2)

    cv2.imwrite('stick_path_viz.jpg', init_frame)


#write the audio based on the contacted frames
def create_audio(frames, contact_index, audio_path):
    total_length = frames * 50    #length of the video in millseconds

    #each frame is 50ms(20fps), drum audio is 200ms
    drum_hit = AudioSegment.from_wav(audio_path)
    if len(drum_hit) != 200:
        drum_hit = drum_hit[:200]

    silent_hit = AudioSegment.silent(duration=200)

    #the audio that matches with the video
    out_audio = AudioSegment.silent(duration=0)

    for i in range(0, len(frames), 4):    #4 frames per 1 interval(either silent or drum hit)
        if i in contact_index or i + 1 in contact_index or i + 2 in contact_index or i + 3 in contact_index:
            out_audio = out_audio + drum_hit
        else:
            out_audio = out_audio + silent_hit

    return out_audio
    
    

if __name__ == '__main__':
    #capture_first_frame()
    tracking(True)
