import cv2
import mediapipe as mp
import numpy as np
import pickle
import sys
from pathlib import Path

# Lunghezza fissa del vettore, DEVE essere identica a quella in train.py
FEATURE_VECTOR_LENGTH = 84

def load_model(model_file="gesture_model.pkl"):
	"""Carica il modello .pkl addestrato."""
	if not Path(model_file).exists():
		print(f"❌ Errore: File modello '{model_file}' non trovato.")
		print("➡️ Esegui prima 'python train.py' per crearlo.")
		sys.exit(1)
		
	try:
		with open(model_file, "rb") as f:
			model = pickle.load(f)
		print(f"✅ Modello '{model_file}' caricato.")
		return model
	except Exception as e:
		print(f"❌ Errore durante il caricamento del modello: {e}")
		sys.exit(1)

def recognize_gestures(model_file="gesture_model.pkl"):
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("❌ Errore: Webcam non accessibile.")
		return

	#CARICA IL MODELLO
	model = load_model(model_file)

	mp_hands = mp.solutions.hands
	layer_width, layer_height = 300, 300
	padding = 7

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

			live_landmarks = {}
			gesture_text = "Nessun gesto"
			color = (0, 0, 255)

			if results.multi_hand_landmarks and results.multi_handedness:
				for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
					label = handedness.classification[0].label  # "Right" o "Left"

					# (Questa parte di normalizzazione rimane identica)
					xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
					ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
					x_min = max(min(xs) - padding, 0)
					x_max = min(max(xs) + padding, w)
					y_min = max(min(ys) - padding, 0)
					y_max = min(max(ys) + padding, h)

					if x_max - x_min == 0 or y_max - y_min == 0:
						continue

					landmarks_layer = []
					for lm in hand_landmarks.landmark:
						x_pixel = (lm.x * w - x_min) * (layer_width / (x_max - x_min))
						y_pixel = (lm.y * h - y_min) * (layer_height / (y_max - y_min))
						landmarks_layer.append((x_pixel / layer_width, y_pixel / layer_height))

					live_landmarks[label] = landmarks_layer
					mp.solutions.drawing_utils.draw_landmarks(frame_draw, hand_landmarks, mp.hands.HAND_CONNECTIONS)

				
				#LOGICA DI PREDIZIONE DEL MODELLO Random Forest
				if live_landmarks:
					# 1. Prepara il vettore di 84 features, come in train.py
					live_vector = np.zeros(FEATURE_VECTOR_LENGTH)
					
					if "Right" in live_landmarks:
						right_data = np.array(live_landmarks["Right"]).flatten()
						live_vector[0:len(right_data)] = right_data
					
					if "Left" in live_landmarks:
						left_data = np.array(live_landmarks["Left"]).flatten()
						live_vector[42:42 + len(left_data)] = left_data
					
					# 2. Chiedi al modello di predire
					prediction = model.predict([live_vector])
					
					# 3. Ottieni il risultato
					gesture_text = f"Gesto: {prediction[0]}"
					color = (0, 255, 0) # Verde = Riconosciuto

			cv2.putText(frame_draw, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
			cv2.imshow("Camera", frame_draw)

			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	recognize_gestures()