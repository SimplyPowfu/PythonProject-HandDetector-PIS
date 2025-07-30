import numpy as np
import cv2 as cv
from pathlib import Path

def get_image():
	Class = 'iloveu'
	Path('DATASET/'+Class).mkdir(parents=True, exist_ok=True)
	cap = cv.VideoCapture(0)
	if not cap.isOpened():
		print("❌ Errore: Webcam non accessibile.")
		return
	i = 0
	while True:
	   
		ret, frame = cap.read()
		frame = cv.flip(frame,1)
		
		if cv.waitKey(1) == ord('c'):
			i+= 1
			cv.imwrite('DATASET/'+Class+'/'+str(i)+'.jpeg',frame)
		cv.imshow('frame', frame)
		if cv.waitKey(1) == ord('q'):
			break
  
	cap.release()
	cv.destroyAllWindows()
if __name__ == "__main__":
   get_image()