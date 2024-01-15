import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
contatore = 0
cam = cv2.VideoCapture(0)
class Mano():
    def __init__(self, tipoMano):               #Inizializza la classe
        self.fingers = [0] * 5
        self.tipoMano = tipoMano
        self.visualizzata = False
    def setFingers(self, posFinger, alzato):    #per settare se il dito è alzato o abbassato
        self.fingers[posFinger]=alzato
    def setVisualizato(self, visualizzata: bool):
        self.visualizzata = visualizzata

manoDestra = Mano("Right")
manoSinistra = Mano("Left")

def getResult(results, manoSinistra, manoDestra, frame, mp_drawing):
    
    if results.multi_hand_landmarks:
            
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            manoDestra = getMano(manoDestra, handedness, hand_landmarks)
            manoSinistra = getMano(manoSinistra, handedness, hand_landmarks)
        #    if results.multi_hand_landmarks:   #cerchi delle dita
        #        for hand_landmarks in results.multi_hand_landmarks:
        #            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        #            fingertip_landmarks = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]
        #            fingertip_px = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in fingertip_landmarks]

        #            for px in fingertip_px:
        #                cv2.circle(frame, px, 35, (0, 255, 0), 0)
    return manoSinistra, manoDestra, frame
def getMano(mano : Mano, handedness, hand_landmarks):
        

    tip_ids = [4, 8, 12, 16, 20]
    if handedness.classification[0].label == mano.tipoMano:
        mano.visualizzata = True
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 2].x:
            if mano.tipoMano == "Left":
                mano.setFingers(0, 0) #pollice sinistro
            else: 
                mano.setFingers(0, 1) #pollice destro
        else: 
            if mano.tipoMano == "Left":
                mano.setFingers(0, 1) #pollice destro
            else: 
                mano.setFingers(0, 0) #pollice sinistro
            
        for finger_id in range(1, 5):  # Inizia da 1 per escludere il pollice
            if hand_landmarks.landmark[tip_ids[finger_id]].y < hand_landmarks.landmark[tip_ids[finger_id] - 2].y:
                mano.setFingers(finger_id, 1)  # Dito alzato
            else:
                mano.setFingers(finger_id, 0)  #Dito chiuso
    
    return mano
def getNumeri(manoSinistra, manoDestra, frame):
    patternS = manoSinistra.fingers
    patternD = manoDestra.fingers
    if patternD == [0,1,0,0,0] and patternS == [0,0,0,0,0]:
        cv2.putText(frame, "1",(20,40),cv2.FONT_ITALIC,1,(0,255,0),2)
    elif patternD == [0,1,1,0,0] and patternS == [0,0,0,0,0]:
        cv2.putText(frame, "2",(20,40),cv2.FONT_ITALIC,1,(0,255,0),2)
    elif patternD == [1,1,1,0,0] and patternS == [0,0,0,0,0]:
        cv2.putText(frame, "3",(20,40),cv2.FONT_ITALIC,1,(0,255,0),2)
    elif patternD == [0,1,1,1,1] and patternS == [0,0,0,0,0]:
        cv2.putText(frame, "4",(20,40),cv2.FONT_ITALIC,1,(0,255,0),2)
    elif patternD == [1,1,1,1,1] and patternS == [0,0,0,0,0]:
        cv2.putText(frame, "5",(20,40),cv2.FONT_ITALIC,1,(0,255,0),2)
    elif patternD == [0,1,1,1,0] and patternS == [0,0,0,0,0] or patternS == [1,0,0,0,0] and patternD == [1,1,1,1,1]:
        cv2.putText(frame, "6",(20,40),cv2.FONT_ITALIC,1,(0,255,0),2)
    elif patternD == [0,1,1,0,1] and patternS == [0,0,0,0,0] or patternS == [1,1,0,0,0] and patternD == [1,1,1,1,1]:
        cv2.putText(frame, "7",(20,40),cv2.FONT_ITALIC,1,(0,255,0),2)
    elif patternD == [0,1,0,1,1] and patternS == [0,0,0,0,0] or patternS == [1,1,1,0,0] and patternD == [1,1,1,1,1]:
        cv2.putText(frame, "8",(20,40),cv2.FONT_ITALIC,1,(0,255,0),2)
    elif patternD == [0,0,1,1,1] and patternS == [0,0,0,0,0] or patternS == [0,1,1,1,1] and patternD == [1,1,1,1,1]:
        cv2.putText(frame, "9",(20,40),cv2.FONT_ITALIC,1,(0,255,0),2)
    elif patternS == [1,1,1,1,1] and patternD == [1,1,1,1,1]:
        cv2.putText(frame, "10",(20,40),cv2.FONT_ITALIC,1,(0,255,0),2)
 
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        manoSinistra.setVisualizato(False)
        manoDestra.setVisualizato(False)
        retn, frame = cam.read()
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        tasto = cv2.waitKey(1)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        manoSinistra, manoDestra, frame = getResult(results, manoSinistra, manoDestra, frame, mp_drawing)
        if not manoSinistra.visualizzata:
            manoSinistra.fingers = [0] * 5
        if not manoDestra.visualizzata:
            manoDestra.fingers = [0] * 5
        else:
            getNumeri(manoSinistra, manoDestra, frame)
        
        print("Left", f"Fingers: {manoSinistra.fingers}", "\t","Right", f"Fingers: {manoDestra.fingers}")
        cv2.imshow('camera', frame)

        if tasto == ord("q"):
            cv2.destroyAllWindows()
            break
        elif tasto == ord("c"):
            cv2.imwrite(f"./foto/{contatore}.jpeg", frame)
            contatore += 1
        if contatore == 20:
            break

cam.release()