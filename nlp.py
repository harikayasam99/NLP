import os
import speech_recognition as sr
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=2) # Adjust for ambient noise
        print("Please speak now...")
        try:
            audio = recognizer.listen(source, timeout=300, phrase_time_limit=300)
            print("Audio recorded successfully.")
        except sr.WaitTimeoutError:
            print("No speech detected within the time limit.")
            return None
    audio_file = "recorded_audio.wav"
    with open(audio_file, "wb") as f:
        f.write(audio.get_wav_data())
    return audio_file
def transcribe_speech(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        print("Transcribing audio...")
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"Transcribed text: {text}")
        return text
    except sr.UnknownValueError:
        print("Google API couldn't understand the speech.")
        return None
    except sr.RequestError as e:
        print(f"Request error from Google Speech API: {e}")
        return None
def extract_symptoms(text):
    symptoms_keywords = [
    "nauseous", "vomiting", "diarrhea", "stomach cramps", "fever", "headache",
    "fatigue", "dizziness", "chills", "sore throat", "cough", "bloating","stomach","chest"
]
    text_lower = text.lower()
    symptoms = [symptom for symptom in symptoms_keywords if symptom in text_lower]
    return symptoms
data = {
"symptoms": ["fever and cough", "headache and fatigue", "sore throat and cough",
"nausea and vomiting", "fever and chills","stomach pain","chest pain"],
"disease": ["flu", "migraine", "cold", "stomach flu", "malaria" ,"ulcer","heart disease"]
}
df = pd.DataFrame(data)
X = df["symptoms"]
y = df["disease"]
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X, y)
def predict_disease(symptoms):
    prediction = model.predict([symptoms])
    return prediction[0]
def main():
    audio_file = record_audio()
    if audio_file:
        print(f"Audio file saved as {audio_file}. Please listen to it to check quality.")
        text = transcribe_speech(audio_file)
        if text:
            print(f"Transcribed Audio: {text}")
            symptoms = extract_symptoms(text)
            if symptoms:
                print(f"Extracted symptoms: {symptoms}")
                symptoms_text = " and ".join(symptoms)
                disease = predict_disease(symptoms_text)
                print(f"Predicted Disease: {disease}")
            else:
                print("No symptoms extracted.")
        else:
            print("Error in transcription. Try speaking more clearly or check the audio quality.")
    else:
        print("No audio recorded.")
if __name__ == "__main__":
    main()
