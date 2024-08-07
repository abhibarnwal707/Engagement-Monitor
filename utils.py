import torch
from facenet_pytorch import MTCNN
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from engagement_metrics import calculate_attention

# Initialize MTCNN and emotion model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)

model_name = "trpakov/vit-face-expression"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

def detect_emotions(image):
    faces, _ = mtcnn.detect(image)
    
    if faces is not None:
        face_box = faces[0].astype(int)  # Convert face_box to integers
        face = image.crop(face_box)
        
        inputs = extractor(images=face, return_tensors="pt")
        outputs = model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy().tolist()[0]
        id2label = model.config.id2label
        class_probabilities = {id2label[i]: prob for i, prob in enumerate(probabilities)}
        
        attention = calculate_attention(face_box, image)
        
        return face, class_probabilities, attention
    return None, None, 0.0

def create_combined_image(face, class_probabilities, attention):
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    axs[0].imshow(np.array(face))
    axs[0].axis('off')
    
    emotions = list(class_probabilities.keys())
    probs = [class_probabilities[emotion] * 100 for emotion in emotions]
    sns.barplot(y=emotions, x=probs, ax=axs[1], orient='h')
    
    axs[1].set_xlim([0, 100])
    axs[1].set_xlabel('Probability (%)')
    axs[1].set_title('Emotion Probabilities')
    
    axs[2].barh(['Attention'], [attention * 100])
    axs[2].set_xlim([0, 100])
    axs[2].set_xlabel('Attention Level (%)')
    axs[2].set_title('Attention Level')

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img
