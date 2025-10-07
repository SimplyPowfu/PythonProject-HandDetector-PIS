import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import shutil
import json

def remove_gesture():
    gesture_name = input("Inserisci il nome del segno da eliminare: ").strip()
    if not gesture_name.isalpha():
        print("❌ Nome del segno non valido. Usa solo lettere.")
        return remove_gesture()  # Richiama sé stessa
    path = Path(f"DATASET/{gesture_name}")
    if not path.exists():
        print(f"❌ Il gesto '{gesture_name}' non esiste! Inserisci un gesto valido.")
        return remove_gesture()
    shutil.rmtree(path)
    print(f"✅ Gesto '{gesture_name}' eliminato.")

def add_gesture():
    gesture_name = input("Inserisci il nome del segno da registrare: ").strip()
    if not gesture_name.isalpha():
        print("❌ Nome del segno non valido. Usa solo lettere.")
        return add_gesture()
    path = Path(f"DATASET/{gesture_name}")
    if path.exists():
        print("❌ Questo segno esiste già! Inseriscine un altro.")
        return add_gesture()
    get_hand_layer_and_landmarks(gesture_name)

def get_hand_layer_and_landmarks(gesture_name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Errore: Webcam non accessibile.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    layer_width, layer_height = 300, 300
    i = 0
    padding = 7  # piccolo margine intorno alla mano

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            frame_draw = frame.copy()
            layer_img = None
            landmarks_normalized = None

            h, w, _ = frame.shape

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

                # Bounding box mano con padding
                x_min = max(min(xs) - padding, 0)
                x_max = min(max(xs) + padding, w)
                y_min = max(min(ys) - padding, 0)
                y_max = min(max(ys) + padding, h)

                # Ritaglia la mano dal frame
                mano_cropped = frame[y_min:y_max, x_min:x_max]

                if mano_cropped.size == 0:
                    continue  # evita errori se la mano è fuori frame

                # Ridimensiona la mano a 300x300
                layer_img = cv2.resize(mano_cropped, (layer_width, layer_height))

                # Crea il layer verde per estetica
                layer_img_green = np.full((layer_height, layer_width, 3), (0, 255, 0), dtype=np.uint8)

                # Landmark sul layer 300x300
                landmarks_layer = []
                for lm in hand_landmarks.landmark:
                    x_pixel = int((lm.x * w - x_min) * (layer_width / (x_max - x_min)))
                    y_pixel = int((lm.y * h - y_min) * (layer_height / (y_max - y_min)))
                    landmarks_layer.append((x_pixel, y_pixel))

                # Normalizza tra 0 e 1 rispetto al layer 300x300
                landmarks_normalized = [(x / layer_width, y / layer_height) for x, y in landmarks_layer]

                # Disegna connessioni blu
                for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
                    start_point = landmarks_layer[start_idx]
                    end_point = landmarks_layer[end_idx]
                    cv2.line(layer_img_green, start_point, end_point, (255, 0, 0), 2)

                # Disegna punti rossi
                for (x, y) in landmarks_layer:
                    cv2.circle(layer_img_green, (x, y), 5, (0, 0, 255), -1)

                # Mostra landmarks sul frame principale (opzionale)
                mp_drawing.draw_landmarks(frame_draw, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Aggiorna layer_img con quello verde disegnato sopra
                layer_img = layer_img_green

            # Mostra finestre
            cv2.imshow("camera", frame_draw)
            if layer_img is not None:
                cv2.imshow("Hand Layer", layer_img)

            # Salva landmark premendo "c"
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") and landmarks_normalized is not None:
                Path(f"DATASET/{gesture_name}").mkdir(parents=True, exist_ok=True)
                i += 1
                file_path = f"DATASET/{gesture_name}/{i}.json"
                with open(file_path, "w") as f:
                    json.dump(landmarks_normalized, f)
                print(f"✅ Salvati landmark mano in {file_path}")

            elif key == ord("q") or cv2.getWindowProperty("camera", cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        action = input("Vuoi ADD o REMOVE un gesto? ").strip().upper()
        if action in ["ADD", "REMOVE"]:
            break
        print("❌ Devi scrivere ADD o REMOVE!")

    if action == "ADD":
        add_gesture()
    elif action == "REMOVE":
        remove_gesture()
