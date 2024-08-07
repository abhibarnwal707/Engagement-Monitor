import dlib
import numpy as np

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def calculate_attention(face_box, image):
    # Convert image to grayscale
    gray = np.array(image.convert('L'))
    
    # Ensure face_box contains integer values
    left, top, right, bottom = map(int, face_box)
    
    # Detect face landmarks
    rect = dlib.rectangle(left, top, right, bottom)
    shape = predictor(gray, rect)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    
    # Calculate eye aspect ratio for blink detection
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)
    
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0

    # Thresholds for blink detection
    EYE_AR_THRESH = 0.25
    if ear < EYE_AR_THRESH:
        attention = 0.0  # Eyes closed
    else:
        attention = 1.0  # Eyes open
    
    return attention

def calculate_engagement(emotion_log, attention_log):
    # Aggregates emotion and attention data to calculate engagement
    if not emotion_log or not attention_log:
        return 0.0
    engagement_scores = []
    for emotions, attention in zip(emotion_log, attention_log):
        engagement_score = sum(emotions.values()) * attention
        engagement_scores.append(engagement_score)
    return sum(engagement_scores) / len(engagement_scores)

def summarize_session(emotion_log, attention_log):
    if not emotion_log:
        prevailing_emotion = "None"
    else:
        emotions_count = {}
        for emotions in emotion_log:
            for emotion, probability in emotions.items():
                if emotion in emotions_count:
                    emotions_count[emotion] += probability
                else:
                    emotions_count[emotion] = probability
        prevailing_emotion = max(emotions_count, key=emotions_count.get)

    engagement_score = calculate_engagement(emotion_log, attention_log)
    
    summary = {
        "Prevailing Emotion": prevailing_emotion,
        "Average Engagement Score": engagement_score,
        "Attention Log": attention_log,
        "Emotion Log": emotion_log,
    }
    
    return summary
