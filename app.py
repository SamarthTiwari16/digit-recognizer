# main.py
# Required libraries: streamlit, numpy, tensorflow, opencv-python, scipy, streamlit-drawable-canvas
# To run: streamlit run main.py

import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
from scipy import ndimage

# --- App Configuration ---
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🧠",
    layout="wide"
)

# --- Model Loading ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """Load the pre-trained digit recognition model."""
    try:
        model = tf.keras.models.load_model("digit_recognizer_cnn.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Image Preprocessing ---
def preprocess_image(img_data):
    """
    Preprocesses the canvas drawing to match the model's input requirements.
    Steps:
    1. Convert RGBA to Grayscale
    2. Invert colors (from black on white to white on black)
    3. Resize to 20x20 pixels to maintain aspect ratio, then pad to 28x28
    4. Apply a binary threshold to clean up the image
    5. Center the digit in the 28x28 frame using the center of mass
    6. Normalize pixel values to the [0, 1] range
    """
    # 1. Convert from RGBA to Grayscale
    # The canvas returns a 4-channel image (R, G, B, A), we only need one channel.
    img = cv2.cvtColor(img_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    # 2. Invert colors and resize
    # The model was trained on white digits on a black background.
    img = 255 - img
    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)

    # 3. Add padding to make it 28x28
    img = np.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=0)

    # 4. Apply binary threshold
    _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # 5. Center the digit
    try:
        cy, cx = ndimage.center_of_mass(img)
        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)
        M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        img = cv2.warpAffine(img, M, (cols, rows))
    except Exception:
        # If center_of_mass fails (e.g., empty canvas), do nothing
        pass

    # 6. Normalize and reshape for the model
    processed_img = img / 255.0
    return processed_img.reshape(1, 28, 28, 1), img


# --- Streamlit UI ---
st.title("Digit Recognizer")
st.markdown("Draw a single digit from 0 to 9 on the canvas below. The model will predict what you've drawn!")

# Create two columns for a cleaner layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Draw Here")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # Fixed fill color with some opacity
        stroke_width=20,
        stroke_color="black",
        background_color="white",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

# Add control buttons underneath the canvas
predict_button = st.button("Predict", use_container_width=True)
clear_button = st.button("Clear", use_container_width=True)

# The 'Clear' button functionality is handled by re-running the script without the old canvas state
if clear_button:
    # A simple way to clear is to rerun the script.
    # Streamlit's state management will handle the rest.
    st.rerun()

with col2:
    st.header("Prediction")
    if model is None:
        st.warning("Model not loaded. Please check the model path and file integrity.")
    elif predict_button and canvas_result.image_data is not None and np.any(canvas_result.image_data):
        # Preprocess the image from the canvas
        processed_input, display_img = preprocess_image(canvas_result.image_data)

        # Make a prediction
        prediction = model.predict(processed_input)[0]
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.markdown(f"### Predicted Digit: **{predicted_digit}**")
        st.markdown(f"Confidence: **{confidence:.2f}**")

        st.divider()

        # Display the preprocessed image that was fed to the model
        st.image(display_img, caption="Preprocessed Image", width=200)

        # Display the probability distribution as a bar chart
        st.markdown("#### Prediction Probabilities")
        # Create a DataFrame for the bar chart
        import pandas as pd
        prob_df = pd.DataFrame({
            'Digit': list(range(10)),
            'Probability': prediction
        })
        st.bar_chart(prob_df.set_index('Digit'))
    else:
        st.info("Draw a digit and click 'Predict' to see the result.")

