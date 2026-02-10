import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import requests
from streamlit_lottie import st_lottie
import time
import random

# --- PAGE CONFIG ---
st.set_page_config(page_title="RPS: Man vs Machine", page_icon="🤖", layout="wide")

# --- CUSTOM CSS FOR ANIMATION ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    # This matches the filename created by your train_model.py
    return tf.keras.models.load_model('rps_mobile_model.h5')

# --- LOAD ANIMATIONS ---
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_robot = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_w51pcehl.json")
lottie_win = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_touohxv0.json")
lottie_lose = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_k0k8k8.json")

# --- INITIALIZE GAME STATE ---
if 'user_score' not in st.session_state:
    st.session_state.update({'user_score': 0, 'cpu_score': 0, 'round': 1, 'game_over': False})

def reset_game():
    st.session_state.update({'user_score': 0, 'cpu_score': 0, 'round': 1, 'game_over': False})

# --- LOGIC ---
model = load_my_model()
class_names = ['Paper', 'Rock', 'Scissors'] # Match your folder order

def predict_gesture(img, model):
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.LANCZOS).convert("RGB")
    img_array = np.asarray(image) / 255.0
    img_reshape = np.reshape(img_array, (1, 224, 224, 3))
    prediction = model.predict(img_reshape)
    return class_names[np.argmax(prediction)], np.max(prediction)

# --- UI LAYOUT ---
st.title("🤖 Rock Paper Scissors: AI Arena")
st.markdown("### Challenge the MobileNetV2 Neural Network")

if not st.session_state.game_over:
    # Scorecard
    c1, c2, c3 = st.columns(3)
    c1.metric("PLAYER", st.session_state.user_score)
    c2.metric("ROUND", f"{st.session_state.round} / 3")
    c3.metric("AI BOT", st.session_state.cpu_score)
    
    st.divider()

    # Move Input
    img_file = st.file_uploader("Upload your hand gesture...", type=['jpg', 'png', 'jpeg'])

    if img_file:
        col_left, col_right = st.columns(2)
        
        with col_left:
            user_img = Image.open(img_file)
            st.image(user_img, caption="Your Move", width=300)
            user_move, conf = predict_gesture(user_img, model)
            st.write(f"AI thinks you played: **{user_move}** ({conf*100:.1f}%)")

        if st.button("FIGHT!"):
            cpu_move = random.choice(class_names)
            
            with col_right:
                st.write(f"### AI played: **{cpu_move}**")
                st_lottie(lottie_robot, height=200)

            # Winner Logic
            if user_move == cpu_move:
                st.warning("It's a Tie!")
            elif (user_move == 'Rock' and cpu_move == 'Scissors') or \
                 (user_move == 'Paper' and cpu_move == 'Rock') or \
                 (user_move == 'Scissors' and cpu_move == 'Paper'):
                st.success("You win this round!")
                st.session_state.user_score += 1
                st.session_state.round += 1
            else:
                st.error("AI wins this round!")
                st.session_state.cpu_score += 1
                st.session_state.round += 1

            # End Game Check
            if st.session_state.user_score == 2 or st.session_state.cpu_score == 2 or st.session_state.round > 3:
                st.session_state.game_over = True
                st.rerun()

else:
    # --- GAME OVER SCREEN ---
    if st.session_state.user_score > st.session_state.cpu_score:
        st_lottie(lottie_win, height=300)
        st.balloons()
        st.header("🏆 YOU BEAT THE MACHINE!")
    else:
        st_lottie(lottie_lose, height=300)
        st.header("🤖 AI DOMINATION... TRY AGAIN?")
    
    if st.button("Restart Tournament"):
        reset_game()
        st.rerun()