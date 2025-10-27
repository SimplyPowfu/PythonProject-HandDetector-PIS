import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import pickle
import sys

def load_compiled_gestures(file_path="compiled_gestures.dat"):
	try:
		with open(file_path, "rb") as f:
			gestures = pickle.load(f)
		return gestures
	except FileNotFoundError:
		print(f"❌ Errore: File dataset compilato '{file_path}' non trovato.")
		print("➡️ Esegui prima 'python compile_dataset.py' per crearlo.")
		sys.exit(1)
	except Exception as e:
		print(f"❌ Errore durante il caricamento di '{file_path}': {e}")
		sys.exit(1)

#Funzione per il calcolo dell'errore 
def calcola_errore_mse(live_landmarks, ref_landmarks):
	"""
	Calcola l'Errore Quadratico Medio (MSE) tra due liste di landmark.
	Un valore basso significa che sono molto simili.
	"""

	if not live_landmarks or not ref_landmarks:
		return float('inf') # Errore infinito se una lista è vuota
	if len(live_landmarks) != len(ref_landmarks):
		return float('inf') # Errore infinito se non hanno lo stesso n. di punti

	live_arr = np.array(live_landmarks)
	ref_arr = np.array(ref_landmarks)

	# Come calcolo l'errore
	# Calcola la differenza quadratica per ogni coordinata (x, y)
    # (live_arr - ref_arr)**2  -> es. [[(x1-x2)**2, (y1-y2)**2], ...]
    # np.sum(..., axis=1)      -> es. [(x1-x2)**2 + (y1-y2)**2, ...] (distanza euclidea quadrata)
    # np.mean(...)             -> fa la media di queste distanze
	error = np.mean(np.sum((live_arr - ref_arr) ** 2, axis=1))
	
	return error

def recognize_gestures(dataset_file="compiled_gestures.dat"):
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("❌ Errore: Webcam non accessibile.")
		return

	mp_hands = mp.solutions.hands
	layer_width, layer_height = 300, 300
	padding = 7

	gestures = load_compiled_gestures(dataset_file)
	print(f"✅ Dataset compilato caricato: {len(gestures)} gesti trovati.")

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

					landmarks_layer = []
					for lm in hand_landmarks.landmark:
						x_pixel = (lm.x * w - x_min) * (layer_width / (x_max - x_min))
						y_pixel = (lm.y * h - y_min) * (layer_height / (y_max - y_min))
						# Normalizza tra 0 e 1
						landmarks_layer.append((x_pixel / layer_width, y_pixel / layer_height))

					live_landmarks[label] = landmarks_layer
					mp.solutions.drawing_utils.draw_landmarks(frame_draw, hand_landmarks, mp_hands.HAND_CONNECTIONS)
			
			miglior_gesto = None
			miglior_errore = float('inf')

			# SOGLIA: L'errore massimo per considerare un gesto valido.
			# Questo è il valore più importante da REGOLARE.
			# Inizia con un valore basso (es. 0.005) e aumentalo se
			# i gesti corretti non vengono riconosciuti.
			SOGLIA_ERRORE = 0.005 

			if live_landmarks:
				for gesture_name, gesture_samples in gestures.items():
					for ref_sample in gesture_samples:
						
						errore_totale_campione = 0
						mani_richieste = 0
						mani_corrispondenti = 0

						if "Right" in ref_sample:
							mani_richieste += 1
							if "Right" in live_landmarks:
								errore_totale_campione += calcola_errore_mse(live_landmarks["Right"], ref_sample["Right"])
								mani_corrispondenti += 1
							else:
								errore_totale_campione = float('inf')

						if "Left" in ref_sample:
							mani_richieste += 1
							if "Left" in live_landmarks:
								errore_totale_campione += calcola_errore_mse(live_landmarks["Left"], ref_sample["Left"])
								mani_corrispondenti += 1
							else:
								errore_totale_campione = float('inf')

						# Se le mani richieste non ci sono, scarta questo campione
						if mani_corrispondenti < mani_richieste:
							continue

						if "Right" not in ref_sample and "Right" in live_landmarks:
							errore_totale_campione += 0.1 # Penalità (da regolare)
						if "Left" not in ref_sample and "Left" in live_landmarks:
							errore_totale_campione += 0.1 # Penalità (da regolare)

						errore_normalizzato = errore_totale_campione / mani_richieste if mani_richieste > 0 else 0

						# Controlla se questo è il miglior punteggio trovato FINORA
						if errore_normalizzato < miglior_errore:
							miglior_errore = errore_normalizzato
							miglior_gesto = gesture_name
			
			if miglior_gesto is not None and miglior_errore < SOGLIA_ERRORE:
				# (Puoi aggiungere l'errore al testo per fare debug)
				# text = f"Gesto: {miglior_gesto} (Err: {miglior_errore:.4f})"
				text = f"Gesto: {miglior_gesto}"
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