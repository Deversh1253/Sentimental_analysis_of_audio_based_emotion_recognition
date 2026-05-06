# ============================================================
# Whisper Tiny — Local PC Testing with Microphone (FIXED)
# Press R to record 5 seconds, Q to quit
# ============================================================
# pip install transformers torch sounddevice numpy scipy

import numpy as np
import sounddevice as sd
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ✏️ Point to the parent folder, NOT the checkpoint subfolder
# The processor files (preprocessor_config.json) live in the parent
MODEL_DIR       = r'C:\\Users\\adhit\\Downloads\\last_project\\whisper_tiny_finetuned-20260505T103838Z-3-001\\whisper_tiny_finetuned'
CHECKPOINT_DIR  = MODEL_DIR + r'\checkpoint-120'   # model weights from best checkpoint

SAMPLE_RATE = 16000
DURATION    = 5  # seconds

# ── Load processor from parent, model weights from checkpoint ─
print('Loading processor...')
processor = WhisperProcessor.from_pretrained(MODEL_DIR)

print('Loading model from checkpoint...')
model = WhisperForConditionalGeneration.from_pretrained(CHECKPOINT_DIR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = model.to(device)
model.eval()

# Force English transcription
model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language='english', task='transcribe'
)

print(f'✅ Model ready on {device}')
print('\n── Controls ──────────────────')
print('  R  →  Record 5 seconds')
print('  Q  →  Quit')
print('──────────────────────────────\n')

# ── Transcribe function ──────────────────────────────────────
def transcribe(audio_np: np.ndarray) -> str:
    # Normalize audio
    if audio_np.max() > 1.0:
        audio_np = audio_np / 32768.0

    inputs = processor(
        audio_np,
        sampling_rate=SAMPLE_RATE,
        return_tensors='pt'
    )
    input_features = inputs.input_features.to(device)

    print(f'   [debug] input shape : {input_features.shape}')
    print(f'   [debug] audio max   : {audio_np.max():.4f}  min: {audio_np.min():.4f}')

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language='en',
            task='transcribe',
            max_new_tokens=225
        )

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    print(f'   [debug] raw output  : "{text}"')
    return text

# ── Main loop ────────────────────────────────────────────────
while True:
    cmd = input('Enter command (R=Record, Q=Quit): ').strip().lower()

    if cmd == 'q':
        print('Bye!')
        break

    elif cmd == 'r':
        print(f'🎙️  Recording for {DURATION} seconds... Speak now!')
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        audio_np = audio.flatten()

        print('⏳ Transcribing...')
        text = transcribe(audio_np)

        if text:
            print(f'📝 Transcription: {text}\n')
        else:
            print('⚠️  No transcription output. Try speaking louder or closer to mic.\n')

    else:
        print('Unknown command. Press R to record or Q to quit.\n')
