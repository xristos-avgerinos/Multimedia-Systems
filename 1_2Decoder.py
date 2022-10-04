import pickle
from functions import *
import cv2

frames = np.fromfile("encoded_frames.bin",  dtype='int8')

attributes = np.fromfile("attributes.bin", dtype='int64')

# Change the shape of the numpy array to a 3D shape with the dimensions = (number of frames, video height, video width )
frames = np.reshape(frames, (attributes[0], attributes[1], attributes[2]))

# Get the movement blocks and the predicted locations for the reconstruction of the p image
vectors = open("Mov_vectors.bin", "rb")
codec = pickle.load(vectors)
vectors.close()

# Create the output file
output = cv2.VideoWriter('videos/catDecoded.avi', cv2.VideoWriter_fourcc(*'MJPG'), attributes[3], (attributes[2], attributes[1]), False)
# frames -> macroblocks -> add motion vectors to macroblocks -> construct Image -> add image to video
for i in range(len(frames)):
    if i == 0:
        # first frame
        output.write(frames[i])
        current_frame = frames[i]
    else:
        macro_block = to_macro_blocks(16, current_frame)
        macro_block2 = to_macro_blocks(16, current_frame)
        for j in range(len(codec[i-1][1])):
            macro_block2[codec[i-1][1][j]] = macro_block[codec[i-1][0][j]]
        image = create_image(45, 80, np.uint8(np.add(to_macro_blocks(16, frames[i]), macro_block2)))
        output.write(image)
        current_frame = image


output.release()
cv2.destroyAllWindows()
