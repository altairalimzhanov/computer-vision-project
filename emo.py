import cv2, json
from deepface import DeepFace
from playsound import playsound
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np

file = open('logs.txt', 'w+')
face_cascade = cv2.CascadeClassifier("dataset/haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)
q = 0

def showReport():
    data = file.read().splitlines()
    labels, captions, percentage = [], [], []
    for emotion in json.loads(data[0])['emotion']:
        labels.append(emotion)
    values = [0] * 7
    for record in data:
        values[labels.index(json.loads(record)['dominant_emotion'])] += 10
    for x in range(len(values)):
        if values[x] == 0:
            x+=1
            continue
        captions.append(labels[x])
        percentage.append(values[x])
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    ax.pie(percentage, labels = captions,autopct='%1.2f%%')
    plt.savefig('output/'+datetime.now().strftime("%Y%m%d_%H%M%S"))
    plt.show()

def log(data):
    if q > 0:
        with open('logs.txt', 'a') as logs:
            #logs.write(str(q) + '#' + str(data) + '\n')
            logs.write(str(data).replace("\'","\"") + '\n')

while video.isOpened():
    _, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)

    for x,y,w,h in face:
        img = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        try:
            analyze = DeepFace.analyze(frame, actions=["emotion"])
            #print(analyze[0]['dominant_emotion'])
            log(analyze[0])
            #print(analyze[0]['age'])
        except:
            print('FACE IS NOT DETECTED!')
            playsound('questions/noface.mp3')
    
    cv2.imshow('video', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == 32:
        q+=1
        if q <= 3:
            playsound(f'questions/{q}.mp3')
        else:
            playsound('questions/thanks.mp3')
            break

video.release()
showReport()