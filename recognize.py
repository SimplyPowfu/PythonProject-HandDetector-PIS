import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import json

def load_gesture_landmarks(dataset_folder):
    gestures = {}
    for gesture_folder in Path(dataset_folder).glob("*"):
        json_files = list(gesture_folder.glob("*.json"))
        landmarks_list = []
        for file in json_files:
            with open(file, "r") as f:
                landmarks = json.load(f)
                landmarks_list.append(landmarks)
        if landmarks_list:
            gestures[gesture_folder.name] = landmarks_list
    return gestures

def landmarks_match(layer_landmarks, gesture_landmarks_list, tolerance=0.1):
    for gesture_landmarks in gesture_landmarks_list:
        match = True
        for (x1, y1), (x2, y2) in zip(layer_landmarks, gesture_landmarks):
            if abs(x1 - x2) > tolerance or abs(y1 - y2) > tolerance:
                match = False
                break
        if match:
            return True
    return False

def recognize_gestures(dataset_folder="DATASET"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Errore: Webcam non accessibile.")
        return

    mp_hands = mp.solutions.hands
    layer_width, layer_height = 300, 300
    padding = 7

    gestures = load_gesture_landmarks(dataset_folder)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            gesture_found = False
            class_name = None

            h, w, _ = frame.shape

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

                x_min = max(min(xs) - padding, 0)
                x_max = min(max(xs) + padding, w)
                y_min = max(min(ys) - padding, 0)
                y_max = min(max(ys) + padding, h)

                # Crea layer verde (solo per calcolo)
                mano_img = np.full((y_max - y_min, x_max - x_min, 3), (0, 255, 0), dtype=np.uint8)
                landmarks_xy = [(int(lm.x * w) - x_min, int(lm.y * h) - y_min) for lm in hand_landmarks.landmark]

                # Centro della mano
                cx = (x_max - x_min) / 2
                cy = (y_max - y_min) / 2

                # Landmark centrati
                landmarks_centered = [(x - cx, y - cy) for x, y in landmarks_xy]

                # Disegna connessioni blu
                for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
                    start_point = landmarks_xy[start_idx]
                    end_point = landmarks_xy[end_idx]
                    cv2.line(mano_img, start_point, end_point, (255, 0, 0), 2)
                # Disegna punti rossi
                for (x, y) in landmarks_xy:
                    cv2.circle(mano_img, (x, y), 5, (0, 0, 255), -1)

                # Resize layer a 300x300
                layer_img = cv2.resize(mano_img, (layer_width, layer_height), interpolation=cv2.INTER_LINEAR)

                # Normalizza landmark centrati
                landmarks_normalized = [((x + (layer_width / 2)) / layer_width,
                                         (y + (layer_height / 2)) / layer_height)
                                        for x, y in landmarks_centered]

                # Confronto landmark
                for label, landmark_sets in gestures.items():
                    if landmarks_match(landmarks_normalized, landmark_sets, tolerance=0.1):
                        gesture_found = True
                        class_name = label
                        break

            # Mostra solo la webcam con il testo del gesto
            display_frame = frame.copy()
            if gesture_found:
                cv2.putText(display_frame, f"Gesto: {class_name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Nessun gesto", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Camera", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_gestures()