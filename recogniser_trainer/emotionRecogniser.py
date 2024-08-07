import cv2, argparse, time, glob, os, sys, subprocess, Update_Model

import socket

UDP_IP = "127.0.0.1"
UDP_PORT1 = 5007
UDP_PORT2 = 5008

#Define variables and load classifier
camnumber = 0
video_capture = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

fishface = cv2.face.FisherFaceRecognizer_create()
try:
    fishface.read("trained_emoclassifier.xml")
except:
    print("no trained xml file found, please run program with --update flag first")
    
parser = argparse.ArgumentParser(description="Options for the emotion-recogniser")
parser.add_argument("--update", help="Call to grab new images and update the model accordingly", action="store_true")
args = parser.parse_args()
facedict = {}
emotions = ["angry", "happy", "sad", "neutral"]

facePosition = ''
facesForPrediction = 8

def crop_face(clahe_image, face):
    for (x, y, w, h) in face:
        faceslice = clahe_image[y:y+h, x:x+w]
        faceslice = cv2.resize(faceslice, (350, 350))
    facedict["face%s" %(len(facedict)+1)] = faceslice
    return faceslice

def update_model(emotions):
    print("Model update mode active")
    check_folders(emotions)
    for i in range(0, len(emotions)):
        save_face(emotions[i])
    print("collected images, looking good! Now updating model...")
    Update_Model.update(emotions)
    print("Done!")

def check_folders(emotions):
    for x in emotions:
        if os.path.exists("dataset\\%s" %x):
            pass
        else:
            os.makedirs("dataset\\%s" %x)

def save_face(emotion):
    print("\n\nplease look " + emotion + ". Press enter when you're ready to have your pictures taken")
    input() #Wait until enter is pressed with the input() method
    video_capture.open(camnumber)
    while len(facedict.keys()) < 16:
        detect_face()
    video_capture.release()
    for x in facedict.keys():
        cv2.imwrite("dataset\\%s\\%s.jpg" %(emotion, len(glob.glob("dataset\\%s\\*" %emotion))), facedict[x])
    facedict.clear()

def recognize_emotion():
    predictions = []
    confidence = []
    for x in facedict.keys():
        pred, conf = fishface.predict(facedict[x])
        predictions.append(pred)
        confidence.append(conf)
        if len(predictions) > 0:
            recognized_emotion = emotions[max(set(predictions), key=predictions.count)]

            print("I think you're %s" %recognized_emotion)
            print(facePosition)

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(str(recognized_emotion).encode(), (UDP_IP, UDP_PORT1))
            sock.sendto(facePosition.encode(), (UDP_IP, UDP_PORT2))
    
    facedict.clear()
    


def grab_webcamframe():
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)


    faces = facecascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # cv2.imshow('frame', frame)      #Show current face

        pos = (x, y, w, h)
        global facePosition
        facePosition = str(pos)

    return clahe_image


def detect_face():
    clahe_image = grab_webcamframe()

    face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(face) == 1:
        faceslice = crop_face(clahe_image, face)
        return faceslice

if args.update:
    update_model(emotions)
else:

    while 1:

        if len(facedict) < facesForPrediction:
            detect_face()
        elif(len(facedict) >= facesForPrediction):
            recognize_emotion()
            time.sleep(.5)

        # key = cv2.waitKey(30) & 0xff
        # if key == 27:       #Breaks display window when Escape key(27) is pressed 
        #     break
