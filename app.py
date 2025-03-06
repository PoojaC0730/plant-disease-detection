import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration and title
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)

# Application title and description
st.title("Plant Disease Detection")
st.markdown("""
This application uses a deep learning model to detect diseases in plant leaves.
Upload an image of a leaf, and the model will predict if it's healthy or identify the disease.
""")

# Function to load the model
@st.cache_resource
def load_model(model_name, num_classes):
    """Load the trained model"""
    if model_name == "vgg19":
        model = models.vgg19(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    
    # Load the model weights
    model.load_state_dict(torch.load(
        f"{model_name}_plant_disease.pth", 
        map_location=torch.device('cpu')
    ))
    model.eval()
    return model

# Define the class names
# You'll need to replace this with your actual class names from the dataset
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Create a sidebar for model selection
st.sidebar.title("Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Model", 
    ["densenet121", "resnet34", "vgg19"],
    index=0
)

# Load the selected model
try:
    num_classes = len(class_names)
    model = load_model(model_choice, num_classes)
    st.sidebar.success(f"Model {model_choice} loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to make predictions
def predict(image, model):
    """Make a prediction for the input image"""
    image_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    return predicted.item(), probabilities.tolist()

# Function to display prediction results
def display_prediction(prediction, probabilities, image):
    # Get the predicted class name
    predicted_class = class_names[prediction]
    
    # Display original image and prediction
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Image")
        st.image(image, width=300, caption="Uploaded Image")
    
    with col2:
        st.subheader("Prediction")
        st.markdown(f"**Prediction:** {predicted_class.replace('___', ' - ')}")
        
        # Check if the leaf is healthy or diseased
        if "healthy" in predicted_class:
            st.success("This plant appears to be healthy! 🌿")
        else:
            st.warning("Disease detected! See details below. 🔬")
        
        # Display confidence score
        confidence = probabilities[prediction] * 100
        st.metric("Confidence", f"{confidence:.2f}%")
    
    # Display top 5 predictions with probabilities
    st.subheader("Top 5 Predictions")
    
    # Get indices of top 5 probabilities
    top5_prob, top5_indices = torch.tensor(probabilities).topk(5)
    
    # Create a dictionary for the top predictions
    top_predictions = {
        "Class": [class_names[idx].replace('___', ' - ') for idx in top5_indices],
        "Probability": [f"{prob * 100:.2f}%" for prob in top5_prob]
    }
    
    # Display as a table
    st.table(top_predictions)
    
    # Display a bar chart of top 5 predictions
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(top_predictions["Class"]))
    ax.barh(y_pos, top5_prob.numpy() * 100, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.split(' - ')[-1] for name in top_predictions["Class"]])
    ax.invert_yaxis()
    ax.set_xlabel('Probability (%)')
    ax.set_title('Top 5 Predictions')
    
    st.pyplot(fig)

# Function to provide disease information and treatment suggestions
def display_disease_info(prediction):
    disease_name = class_names[prediction]
    
    # Skip if healthy
    if "healthy" in disease_name:
        st.success("Your plant is healthy! Continue with regular care and maintenance.")
        return
    
    st.subheader("Disease Information & Treatment")
    
    # Create a dictionary with disease information
    # You can expand this with more accurate information
    disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial spot is a disease that causes dark, water-soaked lesions on leaves and fruit of bell peppers.",
        "treatment": "• Remove and destroy infected plants\n• Apply copper-based bactericides\n• Practice crop rotation to reduce bacterial load",
        "prevention": "• Space plants for good air circulation\n• Avoid overhead watering\n• Use disease-resistant varieties"
    },
    "Pepper__bell___healthy": {
        "description": "Healthy bell pepper plants show no signs of disease and are growing vigorously.",
        "treatment": "• Maintain proper watering and fertilization\n• Monitor for pests regularly",
        "prevention": "• Ensure optimal growing conditions\n• Regularly inspect plants for early signs of disease"
    },
    "Potato___Early_blight": {
        "description": "Early blight is a fungal disease characterized by dark, concentric rings on leaves, leading to leaf drop.",
        "treatment": "• Apply fungicides at the first sign of symptoms\n• Remove and destroy infected plant debris",
        "prevention": "• Rotate crops annually\n• Space plants to improve air circulation\n• Avoid overhead irrigation"
    },
    "Potato___Late_blight": {
        "description": "Late blight is a serious fungal disease that causes dark, oily spots on leaves and can lead to tuber rot.",
        "treatment": "• Apply fungicides at the first sign of blight\n• Remove infected plants immediately",
        "prevention": "• Plant resistant varieties\n• Monitor weather conditions for high humidity and cool temperatures"
    },
    "Potato___healthy": {
        "description": "Healthy potato plants are robust with green foliage and no visible signs of disease.",
        "treatment": "• Continue regular care including watering and fertilization",
        "prevention": "• Ensure proper soil health\n• Inspect regularly for pests or diseases"
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial spot causes dark, water-soaked lesions on tomato leaves and fruit.",
        "treatment": "• Remove infected plants\n• Apply copper-based bactericides as needed",
        "prevention": "• Provide adequate spacing for airflow\n• Avoid working among wet plants"
    },
    "Tomato_Early_blight": {
        "description": "Early blight leads to dark spots on leaves with concentric rings, causing premature leaf drop.",
        "treatment": "• Apply fungicides at the first sign of symptoms\n• Remove affected leaves",
        "prevention": "• Rotate crops each year\n• Space plants properly for good air circulation"
    },
    "Tomato_Late_blight": {
        "description": "Late blight is a devastating fungal disease that can cause rapid decay of leaves and fruit.",
        "treatment": "• Apply fungicides immediately upon detection\n• Remove infected plants from the garden",
        "prevention": "• Use resistant varieties\n• Monitor environmental conditions closely"
    },
    "Tomato_Leaf_Mold": {
        "description": "Leaf mold is caused by a fungus that thrives in humid conditions, leading to yellowing leaves and reduced vigor.",
        "treatment": "• Improve air circulation around plants\n• Remove infected leaves",
        "prevention": "• Avoid excessive humidity by watering at the base of the plant\n• Prune lower leaves to improve airflow"
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Septoria leaf spot is a fungal disease that creates small, dark spots with yellow halos on tomato leaves.",
        "treatment": "• Apply fungicides as needed\n• Remove affected leaves promptly",
        "prevention": "• Practice crop rotation\n• Water at the base of the plant to minimize leaf wetness"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Two-spotted spider mites are tiny pests that cause stippling on leaves and can lead to significant damage if not controlled.",
        "treatment": "• Use miticides or insecticidal soap to control infestations\n• Increase humidity around plants to deter mites",
        "prevention": "• Keep plants healthy to withstand pest pressure\n• Regularly inspect for early signs of infestation"
    },
    "Tomato__Target_Spot": {
        "description": "'Target spot' refers to a fungal disease that creates dark spots with concentric rings on tomato leaves.",
        "treatment": "'Apply fungicides as needed; remove infected foliage promptly.",
        'prevention': 'Practice crop rotation; provide adequate spacing between plants.'
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
      'description': 'This virus causes yellowing and curling of tomato leaves, leading to stunted growth.',
      'treatment': 'Remove infected plants; control aphids that spread the virus.',
      'prevention': 'Use virus-resistant varieties; control aphid populations.'
  },
  'Tomato__Tomato_mosaic_virus': {
      'description': 'Tomato mosaic virus causes mottled discoloration on leaves, stunting growth.',
      'treatment': 'Remove infected plants; control aphid populations.',
      'prevention': 'Use resistant varieties; practice good sanitation.'
  },
  'Tomato_healthy': {
      'description': 'Healthy tomato plants exhibit vigorous growth with lush green foliage.',
      'treatment': 'Continue regular care including watering and fertilization.',
      'prevention': 'Ensure optimal growing conditions; monitor for pests regularly.'
  }
}

    
    # Display information if available
    if disease_name in disease_info:
        info = disease_info[disease_name]
        st.markdown(f"**Disease**: {disease_name.replace('___', ' - ')}")
        st.markdown(f"**Description**: {info['description']}")
        st.markdown("**Treatment Recommendations**:")
        st.markdown(info['treatment'])
        st.markdown("**Prevention**:")
        st.markdown(info['prevention'])
    else:
        # Generic advice if specific disease info is not available
        plant_type = disease_name.split('___')[0]
        disease = disease_name.split('___')[1]
        
        st.markdown(f"**Disease**: {disease} on {plant_type}")
        st.markdown("**General Recommendations**:")
        st.markdown("""
        • Remove and destroy infected plant parts
        • Ensure proper spacing between plants for good air circulation
        • Apply appropriate fungicides or pesticides as needed
        • Rotate crops in vegetable gardens
        • Water at the base of plants to keep foliage dry
        • Maintain plant vigor through proper fertilization
        """)
    
    st.info("Note: For accurate diagnosis and treatment, consult with a local agricultural extension service or plant pathologist.")

# File uploader
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

# When a file is uploaded
if uploaded_file is not None:
    # Read and display the image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Make prediction
    try:
        prediction, probabilities = predict(image, model)
        
        # Display prediction results
        display_prediction(prediction, probabilities, image)
        
        # Show disease information
        display_disease_info(prediction)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")

st.sidebar.subheader("About")
st.sidebar.info("""
This app uses a deep learning model trained on the PlantVillage dataset to identify various plant diseases.
\nSupported plants include: Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry and Tomato.
""")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for plant health | Powered by PyTorch and Streamlit")