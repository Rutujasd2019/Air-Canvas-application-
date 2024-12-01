# All the imports go here
import numpy as np
import mediapipe as mp
from collections import deque
from PIL import Image
import pytesseract
import pyttsx3
import cv2




# Giving different arrays to handle colour points of different colour
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]


# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0

#The kernel to be used for dilation purpose
kernel = np.ones((5,5),np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
colorIndex = 0

# Here is code for Canvas setup
paintWindow = np.zeros((471,636,3)) + 255

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


# Initialize the webcam

new_file = True
cap = cv2.VideoCapture(0)
ret = True

while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    

    frame = cv2.rectangle(frame, (40,10), (100,65), (0,0,0), 2)
    frame = cv2.rectangle(frame, (120,10), (180,65), (255,0,0), 2)
    frame = cv2.rectangle(frame, (200,10), (260,65), (0,255,0), 2)
    frame = cv2.rectangle(frame, (280,10), (340,65), (0,0,255), 2)
    frame = cv2.rectangle(frame, (360,10), (420,65), (0,0,0), 2)
    frame = cv2.rectangle(frame, (440,10), (500,65), (0,0,0), 2)
    frame = cv2.rectangle(frame, (520,10), (580,65), (0,0,0), 2)

    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (129, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (202, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (289, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "New", (362, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Save", (449, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Read", (529, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    #frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # # print(id, lm)
                # print(lm.x)
                # print(lm.y)
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0],landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0],landmarks[4][1])
        cv2.circle(frame, center, 3, (0,255,0),-1)
        #print(center[1]-thumb[1])
        if (thumb[1]-center[1]<30):
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 100: # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0

                paintWindow[67:,:,:] = 255
            elif 120 <= center[0] <= 180:
                    colorIndex = 0 # Blue
            elif 200 <= center[0] <= 260:
                    colorIndex = 1 # Green
            elif 280 <= center[0] <= 340:
                    colorIndex = 2 # Red
            elif 360 <= center[0] <= 420:
                    new_file = True
                    print('New File will be created.')
            elif 440 <= center[0] <= 500 :
                cv2.imwrite("painted_frame.jpg", paintWindow)
                print("saved")
                # Replace  with the path to your handwritten image
                image_path = 'painted_frame.jpg'
                # Use Tesseract OCR to extract text from the image
                extracted_text = pytesseract.image_to_string(Image.open(image_path))
                # Path to save the extracted text as a text file
                output_text_file_path = 'extracted_text.txt'
                if new_file:
                    with open(output_text_file_path, 'w') as text_file:
                        text_file.write(extracted_text)
                    print(f'Text file created successfully at {output_text_file_path}')
                    new_file = False
                else:
                    with open(output_text_file_path, 'a') as text_file:
                        text_file.write(extracted_text + ' ')
                    print(f'Text file updated successfully at {output_text_file_path}')

            elif 520 <= center[0] <= 580:
                ### code to audio
                # Replace 'input.pdf' with the path to your txt file
                input_text_file = 'extracted_text.txt'
                # Read the content of the text file
                with open(input_text_file, 'r') as file:
                    text = file.read()
                # Initialize the text-to-speech engine
                engine = pyttsx3.init()
                # Set properties (optional)
                engine.setProperty('rate', 150)  # Speed of speech
                # Convert text to speech
                engine.say(text)
                # Wait for the speech to finish
                engine.runAndWait()

        else :
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
    # Append the next deques when nothing is detected to avois messing up
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1

    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break


# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
 