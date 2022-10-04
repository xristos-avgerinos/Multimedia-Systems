import cv2
from huffmancodec import *

print("Encoding Started!")

# Load video
video = cv2.VideoCapture('videos/video3.mp4')

# Load the first frame of the video
success, first_frame = video.read()
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Set video codec
video_codec = cv2.VideoWriter_fourcc(*'XVID')

# Set up the output
output = cv2.VideoWriter('encoded_video.avi', video_codec, int(video.get(5)), (int(video.get(3)),
                                                                               int(video.get(4))), False)
output.write(first_frame)

frame_num = 0
encoded_frames = []
previous_frame = first_frame

while video.isOpened():
    # Skip the first frame because we load it outside the loop
    if frame_num == 0:
        success, current_frame = video.read()
        frame_num += 1
        continue

    success, current_frame = video.read()
    frame_num += 1

    if not success:
        break

    # Frame to BW
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate difference
    diff = (current_frame - previous_frame)
    # 2d array to list
    x = list(diff.flat)

    # frequency of every letter in each frame
    symb2freq = collections.Counter(x)

    # huffman encoding
    codec = HuffmanCodec.from_frequencies(symb2freq)
    # encode difference
    encoded = codec.encode(diff)

    # Add encoded result to output
    output.write(encoded)

    # The current frame becomes the previous
    previous_frame = current_frame


video.release()
output.release()

print("Video encoding is done!")
'<------------------------------------------------------------------------------------------------------------------->'
print("Video decoding has started")
# Load video
video = cv2.VideoCapture('encoded_video.avi')

# Load the first frame of the video
success, first_frame = video.read()
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Set video codec
video_codec = cv2.VideoWriter_fourcc(*'XVID')

# Set up the output
output = cv2.VideoWriter('decoded_video.avi', video_codec, int(video.get(5)),
                         (int(video.get(3)), int(video.get(4))), False)
output.write(first_frame)

frame_num = 0

previous_frame = first_frame
while video.isOpened():
    # Skip first frame
    if frame_num == 0:
        success, current_frame = video.read()
        frame_num += 1
        continue

    success, current_frame = video.read()
    frame_num += 1

    if not success:
        break

    # Frame to BW
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # 2d array to list
    x = list(current_frame.flat)

    # frequency of every letter in each frame
    symb2freq = collections.Counter(x)

    # Decode each frame
    codec = HuffmanCodec.from_frequencies(symb2freq)
    current_frame = codec.decode(current_frame)

    # Sum frames
    current_frame = previous_frame + current_frame

    previous_frame = current_frame
    output.write(current_frame)

video.release()
output.release()

print("Video released as 'decoded_video.avi'!")

print("Video decoding is done!")
