import os
import cv2 as cv
import numpy as np
from preprocess import *
from CONFIG.config import *
import matplotlib.pyplot as plt

src = np.float32([[577,463],[707,463],[244,688],[1059,688]])
dst = np.float32([[244,200],[950,200], [244,700],[950,700]])

M_vid = cv.getPerspectiveTransform(src, dst)
Minv_vid = cv.getPerspectiveTransform(dst, src)


input_video = os.path.join(TEST_VIDEOS_DIR, 'challenge_video.mp4')
output_video = os.path.join(OUTPUT_VIDEOS_DIR, 'project_video.mp4')

cap = cv.VideoCapture(input_video)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter(output_video, fourcc, 20.0, (1280, 720))

ret, first_frame = cap.read()
first_frame, c, left_fit, right_fit = pipeline(first_frame, first_frame=True)
out.write(first_frame)

cv.imshow('Video', first_frame)
cv.imshow('Binary Combination', c)


while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame, combinend, left_fit, right_fit = pipeline(  frame,
                                                first_frame=True,
                                                left_fit=left_fit,
                                                right_fit=right_fit,
                                                M=M_vid,
                                                Minv=Minv_vid)
        out.write(frame)
        cv.imshow('Video', frame)
        cv.imshow('Binary Combination', combinend)

        if cv.waitKey(1)==27:
            cap.release()
            cv.destroyAllWindows()
            break
    else:
        break