import cv2
from utils import detect_emotions, create_combined_image
from engagement_metrics import summarize_session
from PIL import Image

def process_webcam_feed(stframe, stop=False):
    cap = cv2.VideoCapture(0)
    emotion_log = []
    attention_log = []

    while not stop:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face, class_probabilities, attention = detect_emotions(image)
        
        if face:
            combined_image = create_combined_image(face, class_probabilities, attention)
            stframe.image(combined_image, channels="RGB")
            emotion_log.append(class_probabilities)
            attention_log.append(attention)
        else:
            stframe.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    summary = summarize_session(emotion_log, attention_log)
    return summary
