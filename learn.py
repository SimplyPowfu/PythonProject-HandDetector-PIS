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
		return remove_gesture()
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
	get_hands_layers_and_landmarks(gesture_name)

def get_hands_layers_and_landmarks(gesture_name):
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("❌ Errore: Webcam non accessibile.")
		return

	mp_hands = mp.solutions.hands
	mp_drawing = mp.solutions.drawing_utils

	layer_width, layer_height = 300, 300
	padding = 7
	i = 0

	with mp_hands.Hands(
		min_detection_confidence=0.6,
		min_tracking_confidence=0.5
	) as hands:
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			frame = cv2.flip(frame, 1)
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results = hands.process(frame_rgb)
			frame_draw = frame.copy()

			h, w, _ = frame.shape

			# Dati da salvare
			data_to_save = {}

			# Layer di visualizzazione per destra e sinistra
			mano_destra_img = None
			mano_sinistra_img = None

			if results.multi_hand_landmarks and results.multi_handedness:
				for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
					label = handedness.classification[0].label  # "Right" o "Left"

					xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
					ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

					# Bounding box mano
					x_min = max(min(xs) - padding, 0)
					x_max = min(max(xs) + padding, w)
					y_min = max(min(ys) - padding, 0)
					y_max = min(max(ys) + padding, h)

					if (x_max - x_min) < 30 or (y_max - y_min) < 30:
						continue

					mano_crop = frame[y_min:y_max, x_min:x_max]
					if mano_crop.size == 0:
						continue

					mano_crop_resized = cv2.resize(mano_crop, (layer_width, layer_height))
					mano_green = np.full((layer_height, layer_width, 3), (0, 255, 0), dtype=np.uint8)

					# Calcolo dei landmark su layer 300x300
					landmarks_layer = []
					for lm in hand_landmarks.landmark:
						x_pixel = int((lm.x * w - x_min) * (layer_width / (x_max - x_min)))
						y_pixel = int((lm.y * h - y_min) * (layer_height / (y_max - y_min)))
						landmarks_layer.append((x_pixel, y_pixel))

					# Normalizza [0, 1]
					landmarks_normalized = [(x / layer_width, y / layer_height) for x, y in landmarks_layer]

					# Disegna connessioni blu
					for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
						cv2.line(mano_green, landmarks_layer[start_idx], landmarks_layer[end_idx], (255, 0, 0), 2)

					# Disegna punti rossi
					for (x, y) in landmarks_layer:
						cv2.circle(mano_green, (x, y), 5, (0, 0, 255), -1)

					# Disegna sulla camera
					mp_drawing.draw_landmarks(frame_draw, hand_landmarks, mp_hands.HAND_CONNECTIONS)

					# Salva dati nel dizionario
					data_to_save[label] = landmarks_normalized

					# Assegna layer per la visualizzazione
					if label == "Right":
						mano_destra_img = mano_green
					elif label == "Left":
						mano_sinistra_img = mano_green

			# Mostra finestre
			cv2.imshow("Camera", frame_draw)
			if mano_destra_img is not None:
				cv2.imshow("Mano Destra", mano_destra_img)
			if mano_sinistra_img is not None:
				cv2.imshow("Mano Sinistra", mano_sinistra_img)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("c"):
				Path(f"DATASET/{gesture_name}").mkdir(parents=True, exist_ok=True)
				i += 1
				file_path = f"DATASET/{gesture_name}/{i}.json"
				# Arrotonda e salva
				data_to_save = {hand: [[round(x, 3), round(y, 3)] for x, y in points] for hand, points in data_to_save.items()}
				with open(file_path, "w") as f:
					json.dump(data_to_save, f, indent=2)
				print(f"✅ Salvato gesto combinato in {file_path}")

			elif key == ord("q") or key == 27 or cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
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
