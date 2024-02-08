import cv2
import mediapipe as mp
from pathlib import Path

cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def visualize_landmarks(image, hand_landmarks):
    image_copy = image.copy()
    mp_drawing.draw_landmarks(image_copy, hand_landmarks, mp_hands.HAND_CONNECTIONS)
def landmarks_match(landmarks_list1, landmarks_list2, tolerance=0.2):
    for p1, p2 in zip(landmarks_list1, landmarks_list2):
        if abs(p1.x - p2.x) > tolerance or abs(p1.y - p2.y) > tolerance or abs(p1.z - p2.z) > tolerance:
            return False
    return True
def recognize_gestures(model, dataset_folder):
    class_folders = list(Path(dataset_folder).glob('*'))
    gesture_labels = [folder.name for folder in class_folders]

    landmarks_to_match = {label: [] for label in gesture_labels}
    
    for class_folder in class_folders:
        images = list(class_folder.glob('*.jpeg'))
        for image_path in images:
            img = cv2.imread(str(image_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                landmarks_to_match[class_folder.name].extend(results.multi_hand_landmarks[0].landmark)

    while True:
        _, image = cap.read()
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        gesture_found = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for label, landmarks_list in landmarks_to_match.items():
                    if landmarks_list and landmarks_match(landmarks_list, hand_landmarks.landmark):
                        gesture_found = True
                        class_name = label
                        break

                visualize_landmarks(image, hand_landmarks)

        if gesture_found:
            cv2.putText(image, f'Gesto: {class_name}', (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(image, 'Nessun gesto', (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Hand Gesture Recognition', image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    hands.close()
    cv2.destroyAllWindows()

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    dataset_folder = 'DATASET'
    recognize_gestures(hands, dataset_folder)
