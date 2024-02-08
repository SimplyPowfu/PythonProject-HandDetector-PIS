import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
contatore = 0
cam = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while True:
    retn, frame = cam.read()
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    tasto = cv2.waitKey(1)
    frame.flags.writeable = False
    results = hands.process(frame)
    
    # Draw the hand annotations on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
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