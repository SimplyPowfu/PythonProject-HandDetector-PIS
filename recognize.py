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

def landmarks_match(layer_landmarks, gesture_landmarks_list, tolerance):
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

				x_min = max(min(xs), 0)
				x_max = min(max(xs), w)
				y_min = max(min(ys), 0)
				y_max = min(max(ys), h)

				if x_max - x_min == 0 or y_max - y_min == 0:
					continue  # evita divisioni per zero

				# Ritaglia la mano dal frame
				mano_cropped = frame[y_min:y_max, x_min:x_max]

				# Ridimensiona a 300x300
				layer_img = cv2.resize(mano_cropped, (layer_width, layer_height))

				# Calcola landmark sul layer
				landmarks_layer = []
				for lm in hand_landmarks.landmark:
					x_pixel = int((lm.x * w - x_min) * (layer_width / (x_max - x_min)))
					y_pixel = int((lm.y * h - y_min) * (layer_height / (y_max - y_min)))
					landmarks_layer.append((x_pixel, y_pixel))

				# Normalizza tra 0 e 1 rispetto al layer
				landmarks_normalized = [(x / layer_width, y / layer_height) for x, y in landmarks_layer]

				# Confronto landmark
				for label, landmark_sets in gestures.items():
					if landmarks_match(landmarks_normalized, landmark_sets, tolerance=0.13):#aumentare il valore per aumentare il gap di errore
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