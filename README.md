# AI-ROCK_PAPER_SCISSORS

# 🤖 Man vs. Machine: Rock Paper Scissors AI

An extraordinary, interactive Rock-Paper-Scissors game powered by **Computer Vision** and **Deep Learning**. Challenge a MobileNetV2-based neural network in a "Best of 3" tournament!

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 🚀 Features
* **Deep Learning Brain:** Uses a custom-trained **MobileNetV2** model for high-speed, accurate gesture recognition.
* **Tournament Mode:** Built-in "Best of 3" logic with a dynamic scorecard.
* **Polished UI:** Interactive web interface with Lottie animations and real-time feedback.
* **Confidence Scoring:** See exactly how sure the AI is about your move.

## 🛠️ Tech Stack
* **Core:** Python 3.11
* **Model Architecture:** MobileNetV2 (Transfer Learning)
* **Frameworks:** TensorFlow / Keras
* **Web App:** Streamlit
* **Image Processing:** Pillow, NumPy

## 📂 Project Structure
```text
├── data/               # Training images (Rock, Paper, Scissors)
├── train_model.py      # Script to clean data and train the AI brain
├── app.py              # The Streamlit web application
└── rps_mobile_model.h5 # The trained neural network model
