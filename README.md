-Dentro la cartella del progetto bisogna creare una cartella chiamata "DATASET" per salvare le informazioni che poi verranno usate come database delle gesture del LIS, dentro la cartella andranno messe altre sotto cartelle che l'IA userà per riconoscere i gesti
-Crea un .venv e dopo dal terminale scarica: pip install opencv-python & pip install mediapipe.
P.S.
esegui learn.py per caricare le foto del segno sul database cliccando c, una volta creato il tuo segno dovrai compilare il file binary_dataset.py per convertire la libreria in binario, per provare il riconoscimento numeri esegui handGesture.py e prova, sia con una mano che con due, a alzare e abbassare le dita per comporre i numeri, infine esegui recognize.py, per provare a riconoscere i precedenti segnali dati a learn.py
il programma è in fase di sviluppo. Presenta bug della lettura della mano.


-Inside the project folder, create a folder called ‘DATASET’ to save the information that will then be used as the LIS gesture database. Inside this folder, create other subfolders that the AI will use to recognise gestures.
-Create a .venv and then download from the terminal: pip install opencv-python & pip install mediapipe.
P.S.
Run learn.py to upload the photos of the sign to the database by clicking c. To test number recognition, run handGesture.py and try raising and lowering your fingers to compose numbers with either one or both hands. Finally, run recognise.py to try to recognise the previous signals given to learn.py.
The programme is currently under development. It has bugs in hand reading.
