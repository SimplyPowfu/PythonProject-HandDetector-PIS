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

	padding = 7
	layer_width, layer_height = 300, 300
	i = 0

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

			# Se rileva almeno una mano, prende solo la prima
			if results.multi_hand_landmarks:
				hand_landmarks = results.multi_hand_landmarks[0]

				xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
				ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

				# Bounding box
				x_min = max(min(xs) - padding, 0)
				x_max = min(max(xs) + padding, w)
				y_min = max(min(ys) - padding, 0)
				y_max = min(max(ys) + padding, h)

				# Layer verde
				mano_img = np.full((y_max - y_min, x_max - x_min, 3), (0, 255, 0), dtype=np.uint8)
				landmarks_xy = [(int(lm.x * w) - x_min, int(lm.y * h) - y_min) for lm in hand_landmarks.landmark]

				# Centro della mano
				cx = (x_max - x_min) / 2
				cy = (y_max - y_min) / 2

				# Trasla i punti rispetto al centro del layer
				landmarks_centered = [(x - cx, y - cy) for x, y in landmarks_xy]

				# Disegna connessioni blu
				for connection in mp_hands.HAND_CONNECTIONS:
					start_idx, end_idx = connection
					start_point = landmarks_xy[start_idx]
					end_point = landmarks_xy[end_idx]
					cv2.line(mano_img, start_point, end_point, (255, 0, 0), 2)

				# Disegna punti rossi
				for (x, y) in landmarks_xy:
					cv2.circle(mano_img, (x, y), 5, (0, 0, 255), -1)

				# Resize layer a 300x300
				layer_img = cv2.resize(mano_img, (layer_width, layer_height), interpolation=cv2.INTER_LINEAR)

				# Normalizza punti centrati rispetto a 300x300
				landmarks_normalized = [((x + (layer_width/2)) / layer_width, (y + (layer_height/2)) / layer_height)
										for x, y in landmarks_centered]

				mp_drawing.draw_landmarks(frame_draw, hand_landmarks, mp_hands.HAND_CONNECTIONS)

			# Mostra finestre
			cv2.imshow("camera", frame_draw)
			if layer_img is not None:
				cv2.imshow("Hand Layer", layer_img)

			# Salva landmark in JSON premendo "c"
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
