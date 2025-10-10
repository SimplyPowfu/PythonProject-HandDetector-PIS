import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import json

# Caricamento dataset
def load_gesture_landmarks(dataset_folder):
	gestures = {}
	for gesture_folder in Path(dataset_folder).glob("*"):
		json_files = list(gesture_folder.glob("*.json"))
		landmarks_list = []
		for file in json_files:
			with open(file, "r") as f:
				data = json.load(f)
				# Ogni file contiene {"Right": [...], "Left": [...]} (una o entrambe)
				landmarks_list.append(data)
		if landmarks_list:
			gestures[gesture_folder.name] = landmarks_list
	return gestures

# Funzione per confronto dei landmarks
def landmarks_match(live_landmarks, ref_landmarks, tolerance):
	"""
	live_landmarks e ref_landmarks sono liste [(x, y), ...] normalizzate [0,1].
	Ritorna True se la distanza di ogni punto è sotto la tolleranza.
	"""
	if not live_landmarks or not ref_landmarks:
		return False
	if len(live_landmarks) != len(ref_landmarks):
		return False

	for (x1, y1), (x2, y2) in zip(live_landmarks, ref_landmarks):
		if abs(x1 - x2) > tolerance or abs(y1 - y2) > tolerance:
			return False
	return True

# Funzione principale di riconoscimento
def recognize_gestures(dataset_folder="DATASET"):
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("❌ Errore: Webcam non accessibile.")
		return

	mp_hands = mp.solutions.hands
	layer_width, layer_height = 300, 300
	padding = 7

	gestures = load_gesture_landmarks(dataset_folder)
	print(f"✅ Dataset caricato: {len(gestures)} gesti trovati.")

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

			h, w, _ = frame.shape
			frame_draw = frame.copy()

			# Dizionario con i landmark normalizzati live: {"Right": [...], "Left": [...]}
			live_landmarks = {}

			if results.multi_hand_landmarks and results.multi_handedness:
				for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
					label = handedness.classification[0].label  # "Right" o "Left"

					xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
					ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

					x_min = max(min(xs) - padding, 0)
					x_max = min(max(xs) + padding, w)
					y_min = max(min(ys) - padding, 0)
					y_max = min(max(ys) + padding, h)

					if x_max - x_min == 0 or y_max - y_min == 0:
						continue

					# Conversione in coordinate layer 300x300
					landmarks_layer = []
					for lm in hand_landmarks.landmark:
						x_pixel = (lm.x * w - x_min) * (layer_width / (x_max - x_min))
						y_pixel = (lm.y * h - y_min) * (layer_height / (y_max - y_min))
						landmarks_layer.append((x_pixel / layer_width, y_pixel / layer_height))

					live_landmarks[label] = landmarks_layer
					mp.solutions.drawing_utils.draw_landmarks(frame_draw, hand_landmarks, mp_hands.HAND_CONNECTIONS)

			# Confronto con dataset
			recognized_gesture = None
			tolerance = 0.13  # più alto = più permissivo

			if live_landmarks:
				for gesture_name, gesture_samples in gestures.items():
					for ref in gesture_samples:
						match_right = match_left = True

						# Se nel file c’è la destra, confronta
						if "Right" in ref:
							if "Right" not in live_landmarks:
								match_right = False
							else:
								match_right = landmarks_match(live_landmarks["Right"], ref["Right"], tolerance)

						# Se nel file c’è la sinistra, confronta
						if "Left" in ref:
							if "Left" not in live_landmarks:
								match_left = False
							else:
								match_left = landmarks_match(live_landmarks["Left"], ref["Left"], tolerance)

						# Se entrambi (presenti o meno) combaciano
						if match_right and match_left:
							recognized_gesture = gesture_name
							break

					if recognized_gesture:
						break

			# Output video
			if recognized_gesture:
				text = f"Gesto: {recognized_gesture}"
				color = (0, 255, 0)
			else:
				text = "Nessun gesto"
				color = (0, 0, 255)

			cv2.putText(frame_draw, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
			cv2.imshow("Camera", frame_draw)

			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	recognize_gestures()
