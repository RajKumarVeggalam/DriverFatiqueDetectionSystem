from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import playsound
import threading
import imutils
import time
import dlib
import cv2

alarm_count = 0
display_hazard = False
hazard_start_time = 0

def sound_alarm(path):
    # Play an alarm sound
    playsound.playsound(path)

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

# Define constants for EAR threshold and consecutive frame count
EYE_AR_THRESH = 0.28
EYE_AR_CONSEC_FRAMES = 45

# Initialize the frame counter and alarm status
COUNTER = 0
ALARM_ON = False

# Load dlib's face detector (HOG-based) and then the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# Loop over frames from the video stream
while True:
    # Grab the frame from the threaded video file stream, resize it, and convert it to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    alert_frame = frame.copy()

    # Detect faces in the grayscale frame
    rects = detector(rgb, 0)

    # Loop over the face detections
    for rect in rects:
        # Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(rgb, rect)
        shape = face_utils.shape_to_np(shape)
        # Draw a rectangle around the face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

        # Draw facial landmarks as white dots
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)


        # Extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Compute the convex hull for the left and right eye, then visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


        # Check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # If the eyes were closed for a sufficient number of frames, sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True

                    # Start a thread to have the alarm sound played in the background
                    t = threading.Thread(target=sound_alarm, args=("alarm.wav",))
                    t.deamon = True
                    t.start()
                    alarm_count += 1
                    if alarm_count == 2:
                        display_hazard = True
                        hazard_start_time = time.time()



                # Draw an alarm on the frame
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                
                cv2.putText(alert_frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # Otherwise, the eye aspect ratio is not below the blink threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False

        # Draw the computed eye aspect ratio on the frame to help with debugging and setting the correct eye aspect ratio thresholds and frame counters
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if display_hazard and (time.time() - hazard_start_time) < 600:
            cv2.putText(frame, "Hazard Lights     : ON", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Auto-Deceleration : ON", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if display_hazard and (time.time() - hazard_start_time) < 600:
            cv2.putText(alert_frame, "Hazard Lights     : ON", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(alert_frame, "Auto-Deceleration : ON", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    cv2.imshow("Alert View", alert_frame)
    

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
