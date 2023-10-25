# Import the necessary packages to make the hand-tracking work
import cv2
import mediapipe as mp
import time

# Setup web-cam for capture
cap = cv2.VideoCapture(0)

# Create hands object
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Set the time variables to 0
previousTime = 0
currentTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    # Insert a for loop to check if we have multiple hands and to extract them
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:    # handLms is a single hand
            for id, lm in enumerate(handLms.landmark):

                # Print (id,lm)
                height, width, channel = img.shape

                # Find the position of the center
                cx, cy = int(lm.x*width), int(lm.y*height)
                print(id, cx, cy)

                # Giant circle at the center of the hand
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate the fps of the web-cam
    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    # Show the image to the GUI
    cv2.imshow("Image", img)
    cv2.waitKey(1)
