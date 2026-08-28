# Import third-party libraries
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

# Import custom modules
import drawing


def collect_data_from_vid(videos_paths: list, file_labels: list):

    # create objects for base options, hand landmarker, etc...
    baseOptions = mp.tasks.BaseOptions
    handLandmarker = vision.HandLandmarker
    handLandmarkerOptions = vision.HandLandmarkerOptions
    visionRunningMode = vision.RunningMode

    # store the hand tracking model's path 
    model_path = 'hand_landmarker.task'

    # setting options with video mode
    options = handLandmarkerOptions(
        base_options = baseOptions(model_asset_path= model_path),
        running_mode = visionRunningMode.VIDEO,
        num_hands = 2
    )

    # it will process each video in the parameter list and sign data
    # to each label
    for video, label in zip(videos_paths, file_labels):
        print(f"processing {video} |    class {label}")

        # set up the task
        landmarker = handLandmarker.create_from_options(options)

        # capture the video in the loop
        cap = cv2.VideoCapture(video)

        frame_count = 0
        frame_skip = 3

        # a loop to read frames sequentially from the video file
        while True: 

            # break the loop if the frame is not being read
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            current_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # skip frames to speed up the process
            # without any significant changes to the data
            if frame_count % frame_skip:
                continue

            # turn the colors to RGB (which mediapipe uses) from BGR (which how openCV reads it)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # convert the frame received from openCV to a mediapipe’s Image object
            mp_image = mp.Image(mp.ImageFormat.SRGB, frame)
            video_result = landmarker.detect_for_video(mp_image, current_timestamp_ms)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # call the function to collect data from each video
            frame = drawing.draw_landmarks_on_video(frame, video_result, True, label)

        cap.release()
        landmarker.close()

