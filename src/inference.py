import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# --- 1. SET UP DEVICE ---
# Check if CUDA (GPU support) is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# --- MODEL AND PROCESSOR LOADING ---
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
sampling_rate = 16000  # Define the sampling rate

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

# Load the model and move it to the specified device (GPU or CPU)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)


def load_audio(path, sr=16000):
    """Loads an audio file and resamples it to the specified sampling rate."""
    speech_array, original_sampling_rate = librosa.load(path, sr=sr)
    return np.array(speech_array, dtype=np.float32)


# --- AUDIO FILE PREPARATION ---
# IMPORTANT: Replace this with the path to your audio file
audio_file = "/path/to/your/audio.mp3"
audio = load_audio(audio_file, sr=sampling_rate)

# --- CHUNKING (important for long MP3s) ---
chunk_duration_s = 30  # 30-second chunks
chunk_len = sampling_rate * chunk_duration_s
transcript = []

print("Starting transcription...")
for start in range(0, len(audio), chunk_len):
    chunk = audio[start : start + chunk_len]

    # Process the audio chunk
    inputs = processor(
        chunk, sampling_rate=sampling_rate, return_tensors="pt", padding=True
    )

    # --- 2. MOVE INPUT TENSOR TO THE SAME DEVICE AS THE MODEL ---
    # Get the input_values tensor and move it to the GPU
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        # Pass the GPU tensor to the model
        logits = model(input_values).logits

    # Argmax on the GPU is fast
    predicted_ids = torch.argmax(logits, dim=-1)

    # The processor's decode function expects a CPU tensor
    predicted_sentence = processor.batch_decode(predicted_ids.cpu())[0]
    transcript.append(predicted_sentence)
    print(
        f"Processed chunk, current transcript length: {len(' '.join(transcript))} chars"
    )


final_text = " ".join(transcript)
print("\n--- Final Transcript ---")
print(final_text)
