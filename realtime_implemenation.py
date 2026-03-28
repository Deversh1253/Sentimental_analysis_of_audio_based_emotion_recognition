import speech_recognition as sr
import torch
import librosa
from transformers import AutoModelForCTC, AutoProcessor
import io

# Load model
model = AutoModelForCTC.from_pretrained(r"C:\\Users\\adhit\\Downloads\\Trial_stt\\final_model")
processor = AutoProcessor.from_pretrained(r"C:\\Users\\adhit\\Downloads\\Trial_stt\\final_model")

r = sr.Recognizer()

while True:
    try:
        user_input = input("Type 'r' to record or 'q' to quit: ").lower()

        if user_input == 'q':
            print("Stopping program...")
            break

        if user_input != 'r':
            continue

        with sr.Microphone() as source:
            print("Recording for 10 sec only...")

            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.record(source, duration=10)

        # waveform conversion
        audio_data = audio.get_wav_data()
        speech, _ = librosa.load(io.BytesIO(audio_data), sr=16000)

        # stt
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

        print("You said:", text)

    except Exception as e:
        print("Error:", e)
