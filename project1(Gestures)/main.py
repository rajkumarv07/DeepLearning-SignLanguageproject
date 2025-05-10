import cv2
import mediapipe as mp
import pyttsx3
import time
from collections import deque
from gesture_recognition import recognize_gesture

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

engine = pyttsx3.init()
engine.setProperty('rate', 160)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:

        gesture_buffer = deque(maxlen=15)
        prev_gesture = None
        last_speak_time = 0
        recognized_text = ""

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            gesture = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                gesture = recognize_gesture(hand_landmarks)
                gesture_buffer.append(gesture)
            else:
                gesture_buffer.append(None)

            # Get most frequent gesture in buffer for smoothing
            if len(gesture_buffer) == gesture_buffer.maxlen:
                most_common = max(set(gesture_buffer), key=gesture_buffer.count)
            else:
                most_common = None

            current_time = time.time()
            if most_common and most_common != prev_gesture:
                recognized_text = most_common
                # Speak gesture if 1.5 seconds passed since last speech
                if current_time - last_speak_time > 1.5:
                    engine.say(most_common)
                    engine.runAndWait()
                    last_speak_time = current_time

                prev_gesture = most_common

            # Display text on screen
            cv2.putText(frame, "Recognize gestures: hi, hello, eat, sleep, walk", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Recognized: {recognized_text}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, "Press 'q' to quit, 'c' to clear", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Sign Language Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                recognized_text = ""
                prev_gesture = None
                gesture_buffer.clear()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
    
