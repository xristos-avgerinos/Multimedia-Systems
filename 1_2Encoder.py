import cv2
import pickle
from functions import *

print('Encoding...')
# Load video
video = cv2.VideoCapture('videos/cat.mp4')

# Video attributes
frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
size = (frameWidth, frameHeight)

# Set the output file
output = cv2.VideoWriter('videos/1_2caterror.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size, False)


original_framesList = []
frame_counter = 1
# error frames =  n+1 - n
error_frames = []

# First Frame
success, previous_frame = video.read()
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
error_frames.append(previous_frame)
codec = []
while video.isOpened():

    success, current_frame = video.read()

    if not success:
        break

    # RGB TO GREYSCALE
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    original_framesList.append(np.array(current_frame, dtype='int8'))

    # Create new Hierarchy level with previous and current frame
    Himg1, Himg2 = upper_level(previous_frame, current_frame)

    # Split into macro-blocks the 2 sub-sampled images.
    img_of_macroblocks1 = to_macro_blocks(4, Himg1[2])
    img_of_macroblocks2 = to_macro_blocks(4, Himg2[2])

    # Blocks to be used for the motion compensation
    movement_blocks = []

    # Check which pairs of blocks has motion
    for i in range(len(img_of_macroblocks1)):
        if has_motion(img_of_macroblocks1[i], img_of_macroblocks2[i]):
            movement_blocks.append(i)

    # Return lower level
    img_of_macroblocks1, img_of_macroblocks2, movement_blocks = to_lower_level(movement_blocks, Himg1, Himg2)

    # find max sad value among macro-block neighbors and store its index so we can reconstruct image later
    predicted = []
    for i in range(len(movement_blocks)):
        predicted.append(get_sad(img_of_macroblocks1, img_of_macroblocks2, movement_blocks[i]))
        img_of_macroblocks1[movement_blocks[i]] = img_of_macroblocks1[predicted[i]]

    # Add to the codec list the movement blocks and the according predicted ones to be used when decoding
    codec.append([predicted, movement_blocks])

    width = int(frameWidth / 16)
    height = int(frameHeight / 16)
    error_frame = create_image(height, width, np.uint8(np.subtract(img_of_macroblocks2, img_of_macroblocks1)))
    output.write(error_frame)
    error_frames.append(error_frame.astype('int8'))

    # Add to the frame counter and set the current frame as the previous one
    frame_counter = frame_counter + 1
    previous_frame = current_frame


video.release()
output.release()
cv2.destroyAllWindows()

# Convert python list to numpy array
error_frames = np.array(error_frames, dtype='int8')

# number of frames, video height, video width, frames per second
attributes = np.array([frameCount, frameHeight, frameWidth, fps], dtype='int64')
error_frames.tofile("encoded_frames.bin")
# number of frames, video height, video width, frames per second to binary file
attributes.tofile("attributes.bin")
# move vectors to binary
vectors = open("Mov_vectors.bin", "wb")
pickle.dump(codec, vectors)
vectors.close()
print('Video decoding is done!')


