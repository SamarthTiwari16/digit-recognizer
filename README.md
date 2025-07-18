# Handwritten Digit Recognizer with Drawing Pad

This project demonstrates a deep learning solution for recognizing handwritten digits (0–9) using a Convolutional Neural Network (CNN). It features a Streamlit-based drawing pad that allows users to draw digits on a canvas and receive real-time predictions.

> **Note**: This is a **learning project** created to explore and apply concepts in deep learning, image preprocessing, and web deployment using Streamlit. It's a hands-on implementation aimed at strengthening practical understanding of AI.

---

## Problem Statement:

In today's digital world, automating handwritten input recognition is vital for digitizing legacy data and creating more inclusive interfaces, especially in educational, administrative, and banking systems. Traditional OCR systems often fail with poorly written or stylized numerals.

>  **Objective**: Build an intelligent digit recognition system that can accurately identify hand-drawn digits from user input in real-time and provide a simple interface for anyone to use.

---

## Real-World Use Cases:

-  **Education Tech** – Auto-grading numeric responses in virtual learning apps.
-  **Banking Sector** – Reading numeric fields on handwritten cheques or forms.
-  **Assistive Technologies** – Helping visually impaired or disabled individuals input digits through gestures/drawing.
-  **Form Digitization** – Converting hand-filled numeric fields in government or healthcare forms into digital records.

---

## Tech Stack:

| Component        | Tech Used                          |
|------------------|------------------------------------|
| UI               | Streamlit + streamlit-drawable-canvas |
| Model            | TensorFlow (CNN trained on MNIST)  |
| Image Processing | OpenCV, NumPy, SciPy               |
| Deployment       | Streamlit Community Cloud          |

---

## Features:

- Real-time digit recognition from drawing pad
- Centering and normalization of drawn input for better accuracy
- Lightweight CNN trained on MNIST dataset
- Fully interactive UI with minimal setup
- Can be deployed in browsers with no installation required by end users

---

## Live Demo:

>  Deployed and hosted on **Streamlit Community Cloud**  
>  **Try it now**: [digit-recognizer-samarth-lp.streamlit.app](https://digit-recognizer-samarth-lp.streamlit.app/)
