import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import os 

# --- Configuration ---
IMG_SIZE = 128
# IMPORTANT: This path is relative to app.py when deployed, 
# so 'my_brain_tumor_model.h5' must be in the same folder as app.py
MODEL_PATH = 'my_brain_tumor_model.h5'
CLASS_NAMES = ['No Tumor', 'Meningioma', 'Glioma', 'Pituitary'] # <<-- VERIFY THIS LIST MATCHES YOUR TRAINING LABELS AND ORDER!

# --- Load Model (Cache to run only once) ---
# st.cache_resource is used for loading persistent resources like models
@st.cache_resource
def load_trained_model():
    """Loads the pre-trained Keras model.

    Resolves the model path relative to this file so deployments that change
    the working directory still work. If the model is missing or fails to
    load, show the underlying exception to help debugging and stop the app.
    """
    # Resolve model path relative to this file (app.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path_abs = os.path.join(base_dir, MODEL_PATH)

    if not os.path.exists(model_path_abs):
        st.error(f"Model file not found at: {model_path_abs}\nPlease copy `my_brain_tumor_model.h5` into the application folder.")
        st.stop()

    try:
        model = load_model(model_path_abs)
        return model
    except Exception as e:
        st.error(f"Error loading model at {model_path_abs}: {e}")
        st.stop()

model = load_trained_model()

# --- Prediction Function ---
def predict_tumor(image, model):
    """Preprocesses the image and gets the model prediction."""
    
    # 1. Convert to array and resize
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) 
    
    # 2. Add batch dimension: (128, 128, 3) -> (1, 128, 128, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 3. Apply VGG16 specific preprocessing (Normalization)
    preprocessed_img = preprocess_input(img_batch)
    
    # 4. Predict
    predictions = model.predict(preprocessed_img)
    
    # 5. Get the class label and confidence
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    
    return predicted_class_index, confidence


# --- Streamlit UI ---
st.title("🧠 Brain Tumor Detection using Deep Learning")
st.markdown("Upload an MRI scan for classification.")
st.divider()

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
    
    # 2. Perform Prediction
    st.write("")
    st.header("Diagnosis:")
    
    with st.spinner('Analyzing scan...'):
        predicted_index, confidence = predict_tumor(image, model)
        predicted_class = CLASS_NAMES[predicted_index]
    
    # 3. Output Results
    if predicted_class == 'No Tumor':
        st.success(f"Result: {predicted_class}")
        st.balloons()
    else:
        st.error(f"Result: {predicted_class} DETECTED")
        
    st.metric("Confidence", f"{confidence*100:.2f}%")
    st.markdown("---")