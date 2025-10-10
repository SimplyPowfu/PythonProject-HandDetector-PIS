import cv2
import mediapipe as mp
import numpy as np

class Mano:
	def __init__(self, tipoMano):
		self.tipoMano = tipoMano
		self.visualizzata = False
		self.green_mask = None

	def setVisualizzata(self, val: bool) -> None:
		self.visualizzata = val

# Inizializzazione MediaPipe per Track mani
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cam = cv2.VideoCapture(0)
manoDestra = Mano("Right")
manoSinistra = Mano("Left")

padding = 7
layer_width, layer_height = 300, 300

with mp_hands.Hands(
		min_detection_confidence=0.6,
		min_tracking_confidence=0.5
) as hands:
	while True:
		retn, frame = cam.read()
		if not retn:
			break
		frame = cv2.flip(frame, 1)
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = hands.process(frame_rgb)

		frame_draw = frame.copy()

		manoDestra.setVisualizzata(False)
		manoSinistra.setVisualizzata(False)
		manoDestra.green_mask = None
		manoSinistra.green_mask = None

		h, w, _ = frame.shape

		if results.multi_hand_landmarks and results.multi_handedness:
			for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
				label = handedness.classification[0].label

				xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
				ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
				x_min = max(min(xs) - padding, 0)
				x_max = min(max(xs) + padding, w)
				y_min = max(min(ys) - padding, 0)
				y_max = min(max(ys) + padding, h)

				# Crea layer verde fisso
				mano_img = np.full((layer_height, layer_width, 3), (0, 255, 0), dtype=np.uint8)

				# Landmark normalizzati sul layer 300x300
				landmarks_xy = []
				for lm in hand_landmarks.landmark:
					x_pixel = int((lm.x * w - x_min) * (layer_width / (x_max - x_min)))
					y_pixel = int((lm.y * h - y_min) * (layer_height / (y_max - y_min)))
					landmarks_xy.append((x_pixel, y_pixel))

				# Disegna connessioni blu
				for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
					start_point = landmarks_xy[start_idx]
					end_point = landmarks_xy[end_idx]
					cv2.line(mano_img, start_point, end_point, (255, 0, 0), 2)

				# Disegna punti rossi
				for (x, y) in landmarks_xy:
					cv2.circle(mano_img, (x, y), 5, (0, 0, 255), -1)

				# Assegna alla mano corretta
				if label == "Right":
					manoDestra.setVisualizzata(True)
					manoDestra.green_mask = mano_img
				elif label == "Left":
					manoSinistra.setVisualizzata(True)
					manoSinistra.green_mask = mano_img

				# Disegna landmarks sul frame principale (opzionale)
				mp_drawing.draw_landmarks(frame_draw, hand_landmarks, mp_hands.HAND_CONNECTIONS)

		# Mostra frame e layer verde per ciascuna mano
		cv2.imshow('camera', frame_draw)
		if manoDestra.visualizzata and manoDestra.green_mask is not None:
			cv2.imshow('Mano Destra', manoDestra.green_mask)
		if manoSinistra.visualizzata and manoSinistra.green_mask is not None:
			cv2.imshow('Mano Sinistra', manoSinistra.green_mask)

		tasto = cv2.waitKey(1) & 0xFF
		if tasto == ord("q") or tasto == 27 or cv2.getWindowProperty("camera", cv2.WND_PROP_VISIBLE) < 1:
			break

cam.release()
cv2.destroyAllWindows()
