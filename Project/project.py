import cv2

if __name__ == "__main__":
    video = cv2.VideoCapture("DrumClip10.mp4")
    
    success, frame = video.read()
    cv2.imwrite("./frame.jpg", frame)