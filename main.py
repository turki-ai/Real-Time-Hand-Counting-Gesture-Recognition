# Import third-party libraries
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import tasks
import tensorflow as tf
import cv2
from os.path import isfile
from time import time

# Import custom modules
import drawing
import hand_video_data_collector
import model_generator

# define gesture classes and corresponding training video paths
GESTURE_LABELS = ['None', 'one', 'two', 'three', 'four', 'five']
VIDEOS = ['videos/None.mp4', 'videos/one.mp4','videos/two.mp4','videos/three.mp4','videos/four.mp4','videos/five.mp4']

# base name for the trained model
model_name = 'counting_gestures_model'

# train a new model if the TFLite file does not exist
if not isfile(f'{model_name}.tflite'):
    hand_video_data_collector.collect_data_from_vid(VIDEOS, GESTURE_LABELS)
    model_generator.model_maker(model_name, GESTURE_LABELS)

# initialize TFLite interpreter and allocate tensors
interpreter = tf.lite.Interpreter(model_path = f'{model_name}.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# store the hand tracking model's path 
hand_landmarker_model_path = 'hand_landmarker.task'

# aliases for mediapipe tasks API components
baseOptions = tasks.BaseOptions
handLandmarker = vision.HandLandmarker
handLandmarkerOptions = vision.HandLandmarkerOptions
handLandmarkerResult = vision.HandLandmarkerResult
visionRunningMode = vision.RunningMode

# global variable and callback to store the latest asynchronous detection result
latest_result = None
def print_result(result: handLandmarkerResult, out_image:mp.Image, timestamp: int):
    global latest_result
    latest_result = result

# configure landmarker for live stream mode
options = handLandmarkerOptions(
    base_options = baseOptions(model_asset_path = hand_landmarker_model_path),
    num_hands = 2,
    running_mode = visionRunningMode.LIVE_STREAM,
    result_callback = print_result
)

# instantiate the HandLandmarker task
landmarker = handLandmarker.create_from_options(options)

# initialize webcam capture
cap = cv2.VideoCapture(0)

# main processing loop
while True:
    success, frame = cap.read()
    # break the loop if the frame is not being read
    if not success:
        break

    # convert BGR(opencv format) to RGB (mediapipe format)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    time_stamp = int(time() * 1000)

    # convert the frame received from openCV to a mediapipe’s Image object
    mp_frame = mp.Image(mp.ImageFormat.SRGB, frame)

    # the detect_async method is used for the live streaming mode which helps in
    # asynchronous background processing to maintain a smooth, lag-free frame rate
    landmarker.detect_async(mp_frame, time_stamp)

    # drawing the hand land mark
    frame = drawing.draw_custom_gestures(frame, latest_result, interpreter, GESTURE_LABELS, input_details, output_details)

    # returning the frame colors back to normal
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame = cv2.resize(frame, (0,0), fx= 1.7, fy= 1.7)
    cv2.imshow('wow', frame)

    # breaking the loop once 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()    
cv2.destroyAllWindows()
landmarker.close()