import streamlit as st
import os
import pickle
import numpy as np
import tensorflow as tf
import torch
import cv2
import librosa
import speech_recognition as sr
import tempfile
import soundfile as sf
import time
import string

from deepface import DeepFace
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import (
    BertForSequenceClassification, 
    BertTokenizer, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from peft import PeftModel, PeftConfig
from PIL import Image


# --- RAG & LLAMA3 IMPORTS ---
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow is using GPU")
    except RuntimeError as e:
        print(e)
        
# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(
    page_title="Aura: Cognitive AI Therapist", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Metrics styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
        color: #58a6ff;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #8b949e;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL PATHS ---
BERT_EMOTION_ADAPTER_PATH = "./bert_lora_happy_model" 
BERT_BASE_MODEL = "bert-base-uncased"
SPEECH_MODEL_PATH = 'speech_emotion_cnn_lstm_option1.h5'
BERT_INTENT_MODEL_PATH = './bert_intent_model'

# --- AUDIO CONSTANTS ---
SR = 22050
DURATION = 3.0
SAMPLES = int(SR * DURATION)
N_MFCC = 40
HOP = 512
N_FFT = 2048
MAX_FRAMES = 130

# --- CONFIG DATA ---
id2label_intent = {0: "inform", 1: "question", 2: "directive", 3: "commissive", 4: "expressive"}
ALLOWED_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "love"]

bert_id2label = {
    0: 'sadness', 1: 'happy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'
}

# --- 2. MODEL LOADER ---
@st.cache_resource
def load_all_models():
    models = {}
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models['device'] = device
    except:
        models['device'] = "cpu"

    # A. BERT Intent
    try:
        print("Loading Intent Model...")
        models['bert_intent_model'] = BertForSequenceClassification.from_pretrained(BERT_INTENT_MODEL_PATH)
        models['bert_intent_tokenizer'] = AutoTokenizer.from_pretrained(BERT_INTENT_MODEL_PATH)
        models['bert_intent_model'].to(models['device'])
    except Exception as e:
        print(f"CRITICAL ERROR: Intent Model Failed: {e}")

    # B. Text Emotion (BERT + LoRA)
    try:
        print("Loading Text Emotion Model (BERT LoRA)...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BERT_BASE_MODEL, num_labels=len(bert_id2label), id2label=bert_id2label, label2id={v: k for k, v in bert_id2label.items()}
        )
        models['text_emotion_model'] = PeftModel.from_pretrained(base_model, BERT_EMOTION_ADAPTER_PATH)
        models['text_emotion_model'].to(models['device'])
        models['text_emotion_tokenizer'] = AutoTokenizer.from_pretrained(BERT_BASE_MODEL)
    except Exception as e:
        print(f"Text Emotion Model Error: {e}")

    # C. Speech Emotion (Local)
    try:
        print("Loading Speech Models...")
        models['speech_model_local'] = load_model(SPEECH_MODEL_PATH)
        models['speech_labels_local'] = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    except Exception as e:
        print(f"Speech Model Error: {e}")

    # D. Gemini SDK Client (Text Validation)
    try:
        print("Initializing Gemini Client...")
        models['gemini_client'] = genai.Client(api_key=API_KEY)
        # ADD THIS LINE TO PREVENT THE KEYERROR
        models['sys_instruction'] = "You are Aura, a professional AI psychologist." 
    except Exception as e:
        print(f"Gemini Error: {e}")

    # E. RAG Pipeline & Llama 3
    try:
        print("Loading local Llama 3 and connecting to Chroma DB...")
        models['llama3'] = Ollama(
            model="llama3:8b",
            num_thread=8,      # Set this to the number of physical cores in your CPU
            temperature=0.7,   # Lower temperature can sometimes speed up "deciding" on words
        )
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        persist_directory = "./mentalchat_chroma_db"
        
        if os.path.exists(persist_directory):
            models['vectorstore'] = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            models['retriever'] = models['vectorstore'].as_retriever(search_kwargs={"k": 2})
            print("Successfully connected to Chroma Vector Database!")
        else:
            print("WARNING: Vector DB not found!")
    except Exception as e:
        print(f"RAG/Llama3 Error: {e}")

    # F. Whisper Transcription Model
    try:
        print("Loading Whisper Transcription Model...")
        models['transcriber'] = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0 if torch.cuda.is_available() else -1,generate_kwargs={"language": "en"})
    except Exception as e:
        print(f"Whisper Error: {e}")
        models['transcriber'] = None

    print("All models loaded successfully.")
    return models

models = load_all_models()

# --- 3. LOGIC FUNCTIONS ---

def preprocess_audio(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if len(audio) < SAMPLES: audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else: audio = audio[:SAMPLES]
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC, hop_length=HOP, n_fft=N_FFT)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    if mfcc.shape[1] < MAX_FRAMES: mfcc = np.pad(mfcc, ((0,0),(0, MAX_FRAMES - mfcc.shape[1])))
    else: mfcc = mfcc[:, :MAX_FRAMES]
    return mfcc[np.newaxis, ..., np.newaxis]

def analyze_text(text):
    try:
        # 1. Local BERT Prediction
        inputs = models['text_emotion_tokenizer'](
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(models['device'])
        
        with torch.no_grad():
            outputs = models['text_emotion_model'](**inputs)
            idx = torch.argmax(torch.nn.functional.softmax(outputs.logits, dim=1)).item()
            model_emo = bert_id2label[idx] 
        
        mapper = {'sadness': 'sad', 'anger': 'angry'}
        model_emo = mapper.get(model_emo, model_emo)
        
        return model_emo

    except Exception as critical_e:
        # If it's hitting 'except', this print will tell you WHY
        print(f"CRITICAL ERROR in analyze_text: {critical_e}")
        return "neutral"
    
        
    except:
        return "neutral"
    
def predict_speech_emotion(audio_path):
    if 'speech_model_local' not in models: return "neutral"
    try:
        X = preprocess_audio(audio_path)
        pred = models['speech_model_local'].predict(X, verbose=0)
        local_emo = models['speech_labels_local'][np.argmax(pred)]
    except:
        local_emo = "neutral"

    return local_emo
        
def transcribe_audio(audio_path):
    if 'transcriber' in models and models['transcriber'] is not None:
        try:
            result = models['transcriber'](audio_path)
            return result['text'].strip()
        except Exception as e:
            print(f"Whisper Error: {e}")
            return ""
    else:
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            try: return r.recognize_google(r.record(source))
            except: return ""

def get_intent(text):
    inputs = models['bert_intent_tokenizer'](text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(models['device'])
    with torch.no_grad(): outputs = models['bert_intent_model'](**inputs)
    return id2label_intent.get(torch.argmax(outputs.logits).item(), "unknown")

def get_response(prompt, history, emotion, intent, modality):
    history_text = ""
    for msg in history[-5:]: 
        role = "Student" if msg['role'] == 'user' else "Aura"
        history_text += f"{role}: {msg['content']}\n"
        
    retrieved_context = ""
    try:
        if 'retriever' in models:
            docs = models['retriever'].invoke(prompt)
            retrieved_context = "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        print(f"Retrieval error: {e}")

    # --- UPDATED ACADEMIC/STUDENT FOCUSED SYSTEM PROMPT ---
    system_prompt = f"""
    You are "Aura," a highly empathetic AI psychologist specifically trained to support high school and college students navigating intense academic and social pressures. 

    CRITICAL INSTRUCTION - EMOTION VALIDATION:
    The student's true underlying emotion is: **{emotion.upper()}** (Detected via {modality.upper()}). 
    You MUST prioritize, validate, and gently speak to this emotion. In high-pressure academic environments, students frequently mask their stress, fear, or sadness with sarcasm, humor, or denial. If their words ({intent}) contradict this detected {emotion}, gently look past their words to create a safe, non-judgmental space for their true feelings.

    HOW TO USE REFERENCE THERAPY EXAMPLES (MentalChat16K):
    Below are examples of successful clinical therapy sessions. DO NOT quote, summarize, or parrot this text back to the user. Instead, INTERNALIZE the therapeutic techniques demonstrated in the text—observe how the therapist asks open-ended questions, reframes negative self-talk, or paces the dialogue. Seamlessly adapt and apply those exact psychological mechanisms to the student's current struggle.
    ---
    {retrieved_context}
    ---

    STUDENT WELL-BEING FOCUS:
    Your ultimate goal is to help the student process their emotional weight so they can regain cognitive clarity, resilience, and academic focus. Validate their feelings first, then gently guide them toward grounded, manageable next steps to prevent burnout.

    CONVERSATION HISTORY:
    {history_text}
    
    LATEST STUDENT INPUT:
    {prompt}
    
    RESPONSE GUIDELINES:
    1. Warmly validate the {emotion} immediately, addressing their specific intent ({intent}).
    2. Speak to them as a supportive mentor and safe harbor, never as an authority figure, parent, or demanding teacher.
    3. Keep the tone conversational, highly empathetic, and professional. 
    4. Provide your response in exactly 8 to 10 sentences.
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # We use st.empty() to create a placeholder in the chat
            response_placeholder = st.empty()
            full_response = ""
        
            # .stream() returns a generator that yields chunks
            for chunk in models['llama3'].stream(system_prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
        
            response_placeholder.markdown(full_response) # Final clean output
            return full_response
        except Exception as e:
            return f"Error: {e}"
# --- 4. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "status" not in st.session_state: st.session_state.status = "idle" 
if "pending_data" not in st.session_state: st.session_state.pending_data = {}
if "stats" not in st.session_state: st.session_state.stats = {"intent": "-", "text": "-", "speech": "-", "face": "neutral"}
if "last_processed_audio" not in st.session_state: st.session_state.last_processed_audio = None

# --- 5. SIDEBAR LAYOUT ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80)
    st.title("Aura System")
    st.markdown("### Multimodal Sensor Suite")
    
    with st.expander("📷 Visual Sensor (Camera)", expanded=True):
        img_file = st.camera_input("Capture Expression", label_visibility="hidden")
        current_face_emo = "neutral"
        
        if img_file:
            try:
                bytes_data = img_file.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                # If you switched to yolov8 earlier, make sure detector_backend='yolov8'
                objs = DeepFace.analyze(cv2_img, actions=['emotion'], enforce_detection=False, detector_backend='opencv', silent=True)
                
                current_face_emo = (objs[0] if isinstance(objs, list) else objs)['dominant_emotion']
                st.success(f"Visual Lock: {current_face_emo.upper()}")
            except Exception as e:
                # THIS IS THE CRITICAL CHANGE: We are printing the actual error 'e'
                st.error(f"DeepFace Error: {e}")
                print(f"CRITICAL VISUAL ERROR: {e}")
    
    st.markdown("---")
    st.markdown("### 📊 Session Telemetry")
    m1, m2 = st.columns(2)
    m1.metric("Intent", st.session_state.stats['intent'].title())
    m2.metric("Face", current_face_emo.title())
    
    m3, m4 = st.columns(2)
    m3.metric("Speech", st.session_state.stats['speech'].title())
    m4.metric("Text", st.session_state.stats['text'].title())

    st.markdown("---")
    if st.button("Reset Session", type="primary"):
        st.session_state.messages = []
        st.session_state.stats = {"intent": "-", "text": "-", "speech": "-", "face": "neutral"}
        st.rerun()

# --- 6. MAIN INTERFACE ---
st.title("🧠 V-AI-DHYA: Cognitive Therapist")
st.caption("A safe space powered by Multimodal Emotion AI")

chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"]=="user" else "🧠"):
            st.markdown(message["content"])

if st.session_state.status == "conflict":
    with st.container():
        st.warning("⚠️ **Emotional Dissonance Detected**")
        st.write("I sense a disconnect between your expression and your words.")
        c1, c2 = st.columns(2)
        if c1.button(f"Trust Face ({st.session_state.pending_data['face']})", use_container_width=True):
            r = get_response(st.session_state.pending_data['msg'], st.session_state.messages, st.session_state.pending_data['face'], st.session_state.pending_data['intent'], "visual")
            st.session_state.messages.append({"role": "assistant", "content": r})
            st.session_state.status = "idle"
            st.rerun()
        if c2.button(f"Trust Text ({st.session_state.pending_data['text']})", use_container_width=True):
            r = get_response(st.session_state.pending_data['msg'], st.session_state.messages, st.session_state.pending_data['text'], st.session_state.pending_data['intent'], "text")
            st.session_state.messages.append({"role": "assistant", "content": r})
            st.session_state.status = "idle"
            st.rerun()

# --- INPUT AREA ---
st.markdown("###")
col_audio, col_text = st.columns([1, 4])

with col_audio:
    audio_val = st.audio_input("🎙️ Speak")

if audio_val:
    current_audio_bytes = audio_val.getvalue()
    
    if st.session_state.last_processed_audio != current_audio_bytes:
        st.session_state.last_processed_audio = current_audio_bytes
        
        with st.status("Processing Audio Signal...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(current_audio_bytes)
                tmp_path = tmp.name
            
            status.write("Transcribing audio via Whisper...")
            transcript = transcribe_audio(tmp_path)
            
            if transcript:
                st.session_state.messages.append({"role": "user", "content": f"🎤 {transcript}"})
                status.write("Analyzing acoustic features...")
                speech_emo = predict_speech_emotion(tmp_path)
                status.write("Analyzing linguistic content...")
                text_emo = analyze_text(transcript)
                intent = get_intent(transcript)
                
                final_emo = speech_emo # Voice prioritizes over text natively
                st.session_state.stats = {"intent": intent, "text": text_emo, "speech": speech_emo, "face": current_face_emo}
                
                status.write("Retrieving context & generating Llama 3 response...")
                response = get_response(transcript, st.session_state.messages, final_emo, intent, "speech")
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                os.remove(tmp_path)
                status.update(label="Complete", state="complete", expanded=False)
                st.rerun()
            else:
                status.update(label="Audio Unintelligible", state="error")
                st.error("Could not understand audio.")
                os.remove(tmp_path)

prompt = st.chat_input("Type your thoughts here...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.status("Processing your input...", expanded=True) as status:
        status.write("Analyzing text emotion & intent...")
        text_emo = analyze_text(prompt)
        intent = get_intent(prompt)
        face_emo = current_face_emo
    
        st.session_state.stats = {"intent": intent, "text": text_emo, "speech": "N/A", "face": face_emo}
    
        pos_set = {'happy', 'surprise', 'joy', 'love'}
        neg_set = {'sad', 'angry', 'fear', 'disgust'}
        is_conflict = (text_emo in pos_set and face_emo in neg_set) or (text_emo in neg_set and face_emo in pos_set)
    
        if is_conflict:
            status.update(label="Conflict Detected", state="error", expanded=False)
            st.session_state.status = "conflict"
            st.session_state.pending_data = {"face": face_emo, "text": text_emo, "intent": intent, "msg": prompt}
            st.rerun()
        else:
            status.write("Retrieving context & generating Llama 3 response...")
            final_emo = text_emo if face_emo == 'neutral' else face_emo
            response = get_response(prompt, st.session_state.messages, final_emo, intent, "text")
            st.session_state.messages.append({"role": "assistant", "content": response})
            status.update(label="Complete", state="complete", expanded=False)
            st.rerun()