import json
import pickle
from pathlib import Path
import sys

# (Questa funzione è simile a quella che avevi in recognize.py)
def load_gesture_landmarks(dataset_folder):
	gestures = {}
	print(f"Caricamento dati da: {dataset_folder}")
	
	folder_path = Path(dataset_folder)
	if not folder_path.exists():
		print(f"❌ Errore: La cartella DATASET '{dataset_folder}' non esiste.")
		return None

	for gesture_folder in folder_path.glob("*"):
		if not gesture_folder.is_dir():
			continue
			
		json_files = list(gesture_folder.glob("*.json"))
		landmarks_list = []
		for file in json_files:
			with open(file, "r") as f:
				try:
					data = json.load(f)
					landmarks_list.append(data)
				except json.JSONDecodeError:
					print(f"⚠️ Attenzione: il file {file} è corrotto o vuoto. Ignorato.")
				
		if landmarks_list:
			gestures[gesture_folder.name] = landmarks_list
			print(f"  > Trovato gesto '{gesture_folder.name}' con {len(landmarks_list)} campioni.")
			
	return gestures

def compile_dataset(dataset_folder="DATASET", output_file="compiled_gestures.dat"):
	print("--- Avvio compilazione dataset ---")
	
	# 1. Carica tutti i JSON
	gestures_data = load_gesture_landmarks(dataset_folder)

	if not gestures_data:
		print("❌ Nessun dato trovato. Esegui learn.py")
		sys.exit(1)

	# 2. Salva i dati in un unico file binario
	try:
		with open(output_file, "wb") as f: # "wb" = Write Binary
			pickle.dump(gestures_data, f)
	except Exception as e:
		print(f"❌ Errore durante il salvataggio del file binario: {e}")
		sys.exit(1)
		
	print("\n--- Compilazione completata! ---")
	print(f"✅ Dati salvati in: {output_file}")
	print(f"Gesti totali compilati: {len(gestures_data)}")

if __name__ == "__main__":
	compile_dataset()