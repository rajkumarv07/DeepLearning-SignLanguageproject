# import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def recognize_gesture(hand_landmarks):
    """
    Heuristic gesture recognition for predefined words:
    "hi", "hello", "eat", "sleep", "walk"
    This uses finger positions and landmark comparisons to detect rough gestures.
    Replace this with ML models for improved accuracy.
    """
    landmarks = hand_landmarks.landmark

    # Simple heuristics:
    # Check which fingers are extended (tip landmark y < pip landmark y)
    fingers_extended = []
    finger_tips_pips = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for tip, pip in finger_tips_pips:
        fingers_extended.append(landmarks[tip].y < landmarks[pip].y)
    # Thumb heuristic based on relative x positions
    thumb_extended = landmarks[4].x < landmarks[3].x

    # Gesture: 'hi' = all fingers extended
    if all(fingers_extended) and thumb_extended:
        return "hi"

    # Gesture: 'hello' = only index extended, thumb not extended
    if fingers_extended.count(True) == 1 and fingers_extended[0] and not thumb_extended:
        return "hello"

    # Gesture: 'eat' = all fingers curled including thumb (fingers not extended)
    if not any(fingers_extended) and not thumb_extended:
        return "eat"

    # Gesture: 'sleep' = thumb and pinky extended, others curled
    if fingers_extended[3] and thumb_extended and not fingers_extended[0] and not fingers_extended[1] and not fingers_extended[2]:
        return "sleep"

    # Gesture: 'walk' = middle and ring extended, others curled
    if fingers_extended[1] and fingers_extended[2] and not fingers_extended[0] and not fingers_extended[3]:
        return "walk"

    return None



