# Import third-party libraries
import cv2
import numpy as np
import csv
import math
import random
import os

    # signing each connection in an array
HAND_CONNECTIONS = [
(0, 1),  # Wrist to Thumb CMC
(1, 2),  # Thumb CMC to MCP
(2, 3),  # Thumb MCP to IP
(3, 4),  # Thumb IP to TIP
(0, 5),  # Wrist to Index MCP
(5, 6),  # Index MCP to PIP
(6, 7),  # Index PIP to DIP
(7, 8),  # Index DIP to TIP
(5, 9),  # Index MCP to Middle MCP
(9, 10), # Middle MCP to PIP
(10, 11),# Middle PIP to DIP
(11, 12),# Middle DIP to TIP
(9, 13), # Middle MCP to Ring MCP
(13, 14),# Ring MCP to PIP
(14, 15),# Ring PIP to DIP
(15, 16),# Ring DIP to TIP
(0, 17), # Wrist to Pinky MCP
(13, 17),# Ring MCP to Pinky MCP
(17, 18),# Pinky MCP to PIP
(18, 19),# Pinky PIP to DIP
(19, 20) # Pinky DIP to TIP
]

# defining the drawing function with three parameters
# one for RGB frame, one for detection result, and an optional flag for gestures
def draw_landmarks_on_liveStream(rgb_image, detection_result, isGesture: bool = False):

    # if there isn't a result it will return the frame without changes
    if detection_result is None or not detection_result.hand_landmarks:
        return rgb_image

    # sign the gesture list into variable 
    gesture_list = detection_result.gestures

    # this array contain every point of every hand that is detected
    hand_landmarks_list = detection_result.hand_landmarks

    # handedness contains the classification label (right or left) each detected hand 
    handedness_list = detection_result.handedness

    # signing the frames dimensions to variables
    # third variable is not needed but should be signed
    h_frame, w_frame, _ = rgb_image.shape

    # the loop that responsible for running codes on each detected hand
    for idx in range(len(hand_landmarks_list)):
        gesture = gesture_list[idx] if idx < len(gesture_list) else None
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # storing the pixel coordinates (x, y) for the current hand's landmarks
        pixel_landmarks = []
        for landmark in hand_landmarks:
            px = int(landmark.x * w_frame)
            py = int(landmark.y * h_frame)
            pixel_landmarks.append((px, py))

            # draw a circle on each detected landmark point
            cv2.circle(rgb_image, (px, py), 4 ,(0, 255, 0), -1)

        # drawing a line in each connection
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            cv2.line(rgb_image, pixel_landmarks[start_idx],
                     pixel_landmarks[end_idx], (88, 205, 54), 2)

        # Finding the bounding box coordinates to position the text label above the hand
        x_coordinates = [pt[0] for pt in pixel_landmarks]
        y_coordinates = [pt[1] for pt in pixel_landmarks]
        text_x = int(min(x_coordinates))
        text_y = int(max(min(y_coordinates) - 10, 20))

        # if the isGesture parameter is True and gesture is not none and the gesture
        # is'n None or open palm then this block should clean the gestures' names and
        # display the names or else just display the handedness category name
        if isGesture and gesture and gesture[0].category_name not in ["None", "Open_Palm"]:
            gesture_name = {
              "None": "",
              "Closed_Fist": "Fist",
              "Open_Palm": "Open Palm",
              "Pointing_Up": "Pointing Up",
              "Thumb_Down": "Thumb Down",
              "Thumb_Up": "Thumb Up",
              "Victory": "Victory",
              "ILoveYou": "I Love You"
            }

            raw_gesture_names = gesture[0].category_name

            cv2.putText(rgb_image, f"{gesture_name.get(raw_gesture_names, raw_gesture_names)}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        1, (88, 205, 54), 1, cv2.LINE_AA)

        else:
            cv2.putText(rgb_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        1, (88, 205, 54), 1, cv2.LINE_AA)
        
    return rgb_image

def draw_landmarks_on_video(rgb_image, detection_result, collect_data: bool = False, label: str = ""):

    if detection_result is None or not detection_result.hand_landmarks:
        return rgb_image

    # this array contain every point of every hand that is detected
    hand_landmarks_list = detection_result.hand_landmarks
    if len(hand_landmarks_list) == 0:
        return rgb_image

    # signing the frames dimensions to variables
    # third variable is not needed but should be signed
    h_frame, w_frame, _ = rgb_image.shape

    # the loop that responsible for running codes on each detected hand
    for idx in range(len(hand_landmarks_list)):
        
        hand_landmarks = hand_landmarks_list[idx]

        #if the hand isn't detected the code should move on
        if not hand_landmarks:
            continue

        # storing the pixel coordinates (x, y) for the current hand's landmarks
        pixel_landmarks = []
        for landmark in hand_landmarks:
            px = int(landmark.x * w_frame)
            py = int(landmark.y * h_frame)
            pixel_landmarks.append((px, py))

            # draw a circle on each detected landmark point
            cv2.circle(rgb_image, (px, py), 4 ,(0, 255, 0), -1)

        # drawing a line in each connection
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks):
                cv2.line(rgb_image, pixel_landmarks[start_idx],
                        pixel_landmarks[end_idx], (88, 205, 54), 2)

        # if the parameter collect_data is True then this block should
        # create a folder named csv_folder if it doesn't exist already
        # and collect the coordinates of each detected landmark point
        # from the videos and label each by the label provided by the 
        # parameter and sign each point by their position to the wrist 
        # and augment the data
        if collect_data:
            os.makedirs('csv_folder', exist_ok= True)
            CSV_FILE = f'csv_folder/{label}.csv' if label else 'dataset.csv'
            if not os.path.isfile(CSV_FILE):
                LANDMARK_NAMES = [
                    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
                    "index_mcp", "index_pip", "index_dip", "index_tip",
                    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
                    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
                    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
                ]

                header = ['label'] if label else []
                for name in LANDMARK_NAMES:
                    header.extend([f'{name}_x', f'{name}_y', f'{name}_z'])

                with open(CSV_FILE, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

            wrist_x, wrist_y, wrist_z = hand_landmarks[0].x, hand_landmarks[0].y ,hand_landmarks[0].z
            
            def write_row(coords):
                flat_row = []
                for x, y, z in coords:
                    flat_row.extend([x, y, z])
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if label:
                        writer.writerow([label] + flat_row)
                    else:
                        writer.writerow(flat_row)

            base_coords = [(lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z) for lm in hand_landmarks]

            flipped_coords = [(-x,y,z) for x, y, z in base_coords]

            angle = math.radians(random.uniform(-30,30))
            rotated_coords = [(x* math.cos(angle)- y * math.sin(angle),
                              x* math.sin(angle)+ y * math.cos(angle),z)
                              for x,y,z in base_coords]
            
            scale = random.uniform(0.7, 1.3)
            scaled_coords = [(x * scale, y * scale, z * scale) for x, y, z in base_coords]

            write_row(base_coords)
            write_row(flipped_coords)
            write_row(rotated_coords)
            write_row(scaled_coords)
    return rgb_image

def draw_custom_gestures(rgb_image, detection_result, the_model, gesture_labels, input_details, output_details):

    if detection_result is None or not detection_result.hand_landmarks:
            return rgb_image

    # this array contain every point of every hand that is detected
    hand_landmarks_list = detection_result.hand_landmarks

    # handedness contains the classification label (right or left) each detected hand 
    handedness_list = detection_result.handedness

    # signing the frames dimensions to variables
    # third variable is not needed but should be signed
    h_frame, w_frame, _ = rgb_image.shape

    # prepare variables to use them later
    r_num = 0
    l_num = 0
    label_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}

    # the loop that responsible for running codes on each detected hand
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # storing the pixel coordinates (x, y) for the current hand's landmarks
        pixel_landmarks = []
        for landmark in hand_landmarks:
            px = int(landmark.x * w_frame)
            py = int(landmark.y * h_frame)
            pixel_landmarks.append((px, py))

            # draw a circle on each detected landmark point
            cv2.circle(rgb_image, (px, py), 4 ,(0, 255, 0), -1)

        # drawing a line in each connection
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks):
                cv2.line(rgb_image, pixel_landmarks[start_idx],
                         pixel_landmarks[end_idx], (88, 205, 54), 2)

        # Finding the bounding box coordinates to position the text label above the hand
        x_coordinates = [pt[0] for pt in pixel_landmarks]
        y_coordinates = [pt[1] for pt in pixel_landmarks]
        text_x = int(min(x_coordinates))
        text_y = int(max(min(y_coordinates) - 10, 20))

        # if the_model and gesture_labels are provided as parameter then
        # this block should invoke the model to predict each gesture and
        # write them on the screen alongside the confidence score
        # and write down the number of each gestures in both hands or else
        # it will just write down the handedness category name
        if  the_model and gesture_labels is not None:
            wrist_x, wrist_y, wrist_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z
            base_coords = [(lm.x - wrist_x, lm.y  - wrist_y, lm.z - wrist_z) for lm in hand_landmarks]

            flat_row = []
            for x, y, z in base_coords:
                flat_row.extend([x, y, z])

            input_data = np.array([flat_row], dtype= np.float32)

            the_model.set_tensor(input_details[0]['index'], input_data)
            the_model.invoke()
            predictions = the_model.get_tensor(output_details[0]['index'])
            
            class_index = np.argmax(predictions[0])
            confidence = predictions[0][class_index]
            predicted_label = gesture_labels[class_index]

            if predicted_label != 'None' and confidence > 0.5:

                cv2.putText(rgb_image, f"{predicted_label} ({confidence:.2f})", 
                            (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2)

                current_num = label_to_num.get(predicted_label, 0)

                if handedness[0].category_name == 'Right':
                    r_num = current_num
                
                elif handedness[0].category_name == 'Left':
                    l_num = current_num
                
        else:
            cv2.putText(rgb_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        1, (88, 205, 54), 1, cv2.LINE_AA)

    # this code will calculate the total number of both gestures
    # and it will display it in the meddle of the screen
    if len(hand_landmarks_list) == 2 and l_num > 0 and r_num > 0:
        cv2.putText(rgb_image, f"{l_num + r_num}", 
                    (int(w_frame/2- 30), 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, (0, 255, 0), 2)

    return rgb_image