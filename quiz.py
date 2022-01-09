import cv2
import csv
from cvzone.HandTrackingModule import HandDetector
import cvzone
import time
from twilio.rest import Client
import tkinter as tk
import numpy as np
import mediapipe as mp


##############################################################
from tkinter import *
from tkinter import simpledialog,messagebox



ROOT = tk.Tk()

ROOT.withdraw()
messagebox.showinfo('NOTE','To Get the Result in Whatsapp 💬\nSend message "join planet-in"  On  "+1 (415) 523-8886" ')
# the input dialog
USER_number = simpledialog.askstring(title="Whatsapp Number",
                                  prompt="Whats your Whatsapp Number ? Please Enter Your Country Code eg.+91 XXXXXXXXXX : ")
USER_name = simpledialog.askstring(title="Name",prompt="Whats your Name ? : ")
###################################################################



cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)



class MCQ():
    def __init__(self, data):
        self.question = data[0]
        self.choice1 = data[1]
        self.choice2 = data[2]
        self.choice3 = data[3]
        self.choice4 = data[4]
        self.answer = int(data[5])
 
        self.userAns = None
 
    def update(self, cursor, bboxs):
 
        for x, bbox in enumerate(bboxs):
            x1, y1, x2, y2 = bbox
            if x1 < cursor[0] < x2 and y1 < cursor[1] < y2:
                self.userAns = x + 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
    
# Import csv file data
pathCSV = "Mcqs.csv"
with open(pathCSV, newline='\n') as f:
    reader = csv.reader(f)
    dataAll = list(reader)[1:]
 
# Create Object for each MCQ
mcqList = []
for q in dataAll:
    mcqList.append(MCQ(q))
 
print("Total MCQ Objects Created:", len(mcqList))
 
qNo = 0
qTotal = len(dataAll)
 
 
begin=cv2.imread("begin.jpg")
cv2.imshow("Press any Key to Continue ",begin)
cv2.waitKey(0)
cv2.destroyAllWindows()

game=True
while game:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)
 
    if qNo < qTotal:
        mcq = mcqList[qNo]
 
        img, bbox = cvzone.putTextRect(img, mcq.question, [100, 100], 2, 2,colorT=(240,214,12),font=cv2.FONT_HERSHEY_PLAIN,colorR=(0, 0,102),colorB=(32,53,234), offset=40, border=5)
        img, bbox1 = cvzone.putTextRect(img, mcq.choice1, [100, 250], 2, 2, offset=30, border=4)
        img, bbox2 = cvzone.putTextRect(img, mcq.choice2, [400, 250], 2, 2, offset=30, border=4)
        img, bbox3 = cvzone.putTextRect(img, mcq.choice3, [100, 400], 2, 2, offset=30, border=4)
        img, bbox4 = cvzone.putTextRect(img, mcq.choice4, [400, 400], 2, 2, offset=30, border=4)
        img, bbox5 = cvzone.putTextRect(img, "Life Line", [1050, 250], 2, 2, offset=20, border=3)
        img, bbox6 = cvzone.putTextRect(img, "Quit", [1050, 100], 2, 2, offset=20, border=3)

        if hands:
            lmList = hands[0]['lmList']
            cursor = lmList[8]
            length, info = detector.findDistance(lmList[8], lmList[12])
            print(length)
            if length < 35:
                mcq.update(cursor, [bbox1, bbox2, bbox3, bbox4,bbox5,bbox6])
                print(mcq.userAns)
                if mcq.userAns == 5:
                    lifeline=cv2.imread("lifeline.jpg")
                    cv2.imshow("Sorry ! Im not paid for this ",lifeline)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                if mcq.userAns == 6:
                    quit=cv2.imread("quit.png")
                    cv2.imshow("are u sure ",quit)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    qNo=qTotal+1

                if mcq.userAns is not None:
                    time.sleep(0.3)
                    qNo += 1
    else:
        score = 0
        for mcq in mcqList:
            if mcq.answer == mcq.userAns:
                score += 1
        score = round((score / qTotal) * 100, 2)
        img, _ = cvzone.putTextRect(img, "Quiz Completed", [250, 300], 2, 2, offset=50, border=5)
        img, _ = cvzone.putTextRect(img, f'Your Score: {score}%', [700, 300], 2, 2, offset=50, border=5)
        


    # Draw Progress Bar
    barValue = 150 + (950 // qTotal) * qNo
    cv2.rectangle(img, (150, 600), (barValue, 650), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(img, (150, 600), (1100, 650), (255, 0, 255), 5)
    img, _ = cvzone.putTextRect(img, f'{round((qNo / qTotal) * 100)}%', [1130, 635], 2, 2, offset=16)
    
    cv2.imshow("Img", img)


   

    

 

    if cv2.waitKey(10) & 0xFF == ord('q'):
	    break

####GAME OVER####   
game = False
gameover=cv2.imread("gameover.jpg",cv2.CAP_DSHOW)
cv2.imshow("Gameover! Wellplayed ",gameover)
cv2.waitKey(0)
cv2.destroyAllWindows()

#####GAME OVER####   
    
######################   TWILIO   ##########################################################
    

account_sid = 'AC69b38384513b873060389f558c384f49' 
auth_token = '9f7b26a73c4ad35d14a9e4be50bb88ff' 
client = Client(account_sid, auth_token) 
if score==100:
    message = client.messages.create( 
                                media_url=['https://memetemplateindia.com/wp-content/uploads/2020/05/amitabh-kbc-ek-crore-2-1024x517.jpg'],
                                from_='whatsapp:+14155238886',  
                                body=f'ADBHUT !! {USER_name}.You  are now a millioniare💰💰  !  ',      
                                to=f'whatsapp:{USER_number}' 
                            ) 
else:
    message = client.messages.create( 
                                media_url=['https://memetemplateindia.com/wp-content/uploads/2020/05/kya-kijiyega-is-dhanraashi-ka-e1590870424194.jpeg'],
                                from_='whatsapp:+14155238886',  
                                body=f'Congrats {USER_name}.You  won  ${score}K  !  ',      
                                to=f'whatsapp:{USER_number}' 
                            ) 
print(message.sid)
########################TWILIO#######################################################################
if game == False:
    mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


# Curl counter variables
counter = 0
stage = None

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Visualize angle
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                        255, 255), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == 'down':
                stage = "up"
                counter += 1
                print(counter)
        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(
                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q') or counter == 5:
            break

    cap.release()
    cv2.destroyAllWindows()
#########################   TWILIO  ############################################################ 
######### adbhut ############

# if score==100:
#     win=cv2.VideoCapture("adbhut.mp4")
#     while (True):
#         ret,frame=win.read()
#         frame=cv2.resize(frame,(1200,720))
#         cv2.imshow("output",frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

        
    ######### 1 core ############  
