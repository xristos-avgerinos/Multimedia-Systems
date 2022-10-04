import cv2
import numpy as np

def frame_to_macroblocks(frame, window=32):
    previous_width = frame.shape[0]
    width = fit_size(previous_width)
    previous_height = frame.shape[1]
    height = fit_size(previous_height)

    # Add padding so it can be divided by 32
    width_pad = (0, width - previous_width)
    height_pad = (0, height - previous_height)
    depth_pad = (0, 0)
    padding = (width_pad, height_pad, depth_pad)
    padded_frame = np.pad(frame, padding, mode='constant')
    # Create 32x32 blocks based on the x-y plane of the frame.
    macroblocks = []
    for w in range(0, width - window, window):
        row = []
        for h in range(0, height - window, window):
            macroblock = padded_frame[w:w + window, h:h + window]
            row.append(macroblock)
        macroblocks.append(row)
    return macroblocks

def macroblocks_to_frame(macroblocks):
    # gather all macroblocks in  x  for each row.
    rows = []
    for row in macroblocks:
        rows.append(np.concatenate([macroblock for macroblock in row], axis=1))

    # gather all macroblock in y for each row.
    frame = np.concatenate([row for row in rows], axis=0)
    return frame

def fit_size(x, window=32):
    # Find the closest integer based on x that is divisible by 32.
    return (x + window) - (x % window)


video = cv2.VideoCapture('videos/video4.mp4')

previous = False
mb_prev = None

while video.isOpened():
    success, frame = video.read()

    # Break if the video has ended.
    if not success:
        break

    # Extract macroblocks with size 32
    mb = frame_to_macroblocks(frame, window=32)

    # Keep previous macroblocks except from the first iteration.
    if not previous:
        previous = True
        mb_prev = mb
        continue

    # Replace macroblocks to hide motion.
    for i in range(3, 11):
        mb[i] = mb_prev[i]

    mb_frame = macroblocks_to_frame(mb)

    cv2.imshow('Original Video', frame)
    cv2.imshow('Object removal', mb_frame)

    # Keep previous macro-blocks.
    mb_prev = mb

    cv2.waitKey(30)

video.release()
cv2.destroyAllWindows()
