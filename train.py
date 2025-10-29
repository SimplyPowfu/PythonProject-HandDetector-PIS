#addestratore da JSON usando K-Nearest Neighbors (KNN) o Random Forest (libreria scikit-learn)
#scaricale le librerie di sklearn
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # <-- MODELLO SCELTO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import sys

# Definiamo una lunghezza fissa per il "vettore" di un gesto
# 21 landmark * 2 coordinate (x, y) = 42 features per mano
# 42 (Destra) + 42 (Sinistra) = 84 features totali
FEATURE_VECTOR_LENGTH = 84

def load_compiled_data(file_path="compiled_gestures.dat"):
	"""Carica i dati compilati dal file .dat"""
	if not Path(file_path).exists():
		print(f"❌ Errore: File '{file_path}' non trovato.")
		print("➡️ Esegui prima 'python binary_dataset.py' per crearlo.")
		sys.exit(1)
		
	with open(file_path, "rb") as f:
		data = pickle.load(f)
	return data

def prepare_data(gestures_data):
	"""
	Converte il dizionario di dati in un formato (X, y)
	pronto per scikit-learn.
	"""
	X = []  # Lista dei vettori di features (i dati)
	y = []  # Lista delle etichette (i nomi dei gesti)

	print("Preparazione dati in corso (vettori da 84 features)...")
	
	for gesture_name, samples in gestures_data.items():
		for sample in samples:
			# 1. Crea un vettore vuoto di 84 zeri
			feature_vector = np.zeros(FEATURE_VECTOR_LENGTH)

			if "Right" in sample and sample["Right"]: 
				right_hand_data = np.array(sample["Right"]).flatten()
				feature_vector[0:len(right_hand_data)] = right_hand_data

			if "Left" in sample and sample["Left"]:
				left_hand_data = np.array(sample["Left"]).flatten()
				feature_vector[42:42 + len(left_hand_data)] = left_hand_data

			X.append(feature_vector)
			y.append(gesture_name)

	print(f"Preparazione completata: {len(y)} campioni totali.")
	return np.array(X), np.array(y)

def train_model():
	gestures_data = load_compiled_data()
	if not gestures_data:
		return

	X, y = prepare_data(gestures_data)
	if X.size == 0:
		print("❌ Nessun dato da addestrare. Esegui learn.py e binary_dataset.py.")
		return

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
	print(f"Dati divisi: {len(y_train)} per training, {len(y_test)} per test.")

	print("Addestramento modello Random Forest in corso...")
	# n_estimators = numero di "alberi" nella foresta. 100 è un buon inizio.
	model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 usa tutti i core CPU
	model.fit(X_train, y_train)
	print("✅ Addestramento completato.")

	# 5. Valuta l'accuratezza del modello
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print(f"\n--- Accuratezza del modello (su dati di test): {accuracy * 100:.2f}% ---")
	if accuracy < 0.85:
		print("⚠️ Attenzione: l'accuratezza è migliorabile. Aggiungi più campioni con learn.py,")
		print("   assicurandoti di variare angolazioni e posizioni!")

	# 6. Salva il modello addestrato in un file
	output_file = "gesture_model.pkl"
	with open(output_file, "wb") as f:
		pickle.dump(model, f)

	print(f"\n✅ Modello salvato con successo in: {output_file}")

if __name__ == "__main__":
	train_model()

# https://www.youtube.com/watch?v=w5gB8zyLx-8