import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
@st.cache_resource  # Cache the model to avoid reloading on every prediction
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Specify the path to the model
model = load_model(r"C:\User\plant_classifier_model.h5")  # Update with the correct path

# Get the class names from the model's training data (update with your classes)
class_names =  ['Amaranthus Green', 'Balloon vine', 'Betel Leaves', 'Celery', 'Chinese Spinach', 'Coriander Leaves', 'Curry Leaf', 'Dwarf Copperleaf (Green)']  # Replace with actual class names

# Function to preprocess image for model prediction
def preprocess_image(img, img_size=(224, 224)):
    img = img.resize(img_size)  # Resize the image
    img_array = np.array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the image
    return img_array

# Function to predict the plant class
def predict(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]  # Return the class index

# Streamlit UI
st.title("Plant Classification")
st.write("Upload an image of the plant, and the model will predict its class.")

# Display class names for reference
st.subheader("Model is trained on the following plant classes:")
st.write(", ".join(class_names))

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        with st.spinner('Classifying...'):
            # Preprocess the image for the model
            img_array = preprocess_image(image)
            
            # Make the prediction
            predicted_index = predict(model, img_array)
            
            # Display the prediction
            st.write(f"Prediction: {class_names[predicted_index]}")
