import os
import torch
import torchaudio
from speechbrain import inference
import sounddevice as sd
import scipy.io.wavfile as wav
import time
import numpy as np
from scipy import signal
import threading

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

verification = inference.SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts=device
)

SAMPLE_RATE = 16000
DURATION = 3
THRESHOLD = 0.35
SPEAKER_EMBEDDINGS_FILE = "speaker_embeddings.pt"
MAX_DB_SIZE = 4000

# Watermarking
HIGH_FREQ_MIN = 7000
HIGH_FREQ_MAX = 7800 
WATERMARK_AMPLITUDE = 0.15 # Change this however

if os.path.exists(SPEAKER_EMBEDDINGS_FILE):
    speaker_db = torch.load(SPEAKER_EMBEDDINGS_FILE)
else:
    speaker_db = {}

def monitor_db_size():
    if os.path.exists(SPEAKER_EMBEDDINGS_FILE):
        size = os.path.getsize(SPEAKER_EMBEDDINGS_FILE)
        print(f"Embeddings file size: {size} bytes")

def maybe_evict_oldest_speaker():
    if not os.path.exists(SPEAKER_EMBEDDINGS_FILE):
        return
    size = os.path.getsize(SPEAKER_EMBEDDINGS_FILE)
    if size > MAX_DB_SIZE:
        oldest_user = None
        oldest_time = float('inf')
        for user_id, data in speaker_db.items():
            if data["timestamp"] < oldest_time:
                oldest_time = data["timestamp"]
                oldest_user = user_id
        if oldest_user:
            del speaker_db[oldest_user]
            torch.save(speaker_db, SPEAKER_EMBEDDINGS_FILE)
            print(f"Evicted oldest user: {oldest_user}, DB size now {os.path.getsize(SPEAKER_EMBEDDINGS_FILE)} bytes")

def generate_watermark(duration=DURATION, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    watermark = signal.chirp(t, HIGH_FREQ_MIN, duration, HIGH_FREQ_MAX)
    ## 32767 is for highest 16-bit for PCM
    watermark = (watermark * WATERMARK_AMPLITUDE * 32767).astype(np.int16)
    
    return watermark

def play_watermark():
    watermark = generate_watermark()
    sd.play(watermark, SAMPLE_RATE)
    return watermark

def detect_watermark(audio_data, sample_rate=SAMPLE_RATE):
    sos = signal.butter(10, [HIGH_FREQ_MIN, HIGH_FREQ_MAX], btype='bandpass', fs=sample_rate, output='sos')
    filtered = signal.sosfilt(sos, audio_data)
    
    watermark_energy = np.sum(filtered**2)
    total_energy = np.sum(audio_data**2)
    ratio = watermark_energy / total_energy if total_energy > 0 else 0
    print(f"Watermark ratio: {ratio:.8f}")
    
    # need to tune
    return ratio > 0.0005 

def get_embedding(file_path):
    waveform = verification.load_audio(file_path)
    return verification.encode_batch(waveform).squeeze(1)

def record_audio(filename, duration=DURATION, with_watermark=False):
    print(f"Recording for {duration} seconds. State a command.")
    
    # to avoid when verifying tho should it be continously played?
    if with_watermark:
        watermark_thread = threading.Thread(target=play_watermark)
        watermark_thread.start()
    
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    
    if with_watermark and 'watermark_thread' in locals() and watermark_thread.is_alive():
        watermark_thread.join()
    
    wav.write(filename, SAMPLE_RATE, recording)

def check_uncached_files(command_embedding):
    enrolled_files = [f for f in os.listdir('.') if f.startswith("enrolled_user_") and f.endswith(".wav")]
    for f in enrolled_files:
        user_id = f.replace(".wav", "")
        if user_id not in speaker_db:
            embed = get_embedding(f)
            score = verification.similarity(embed, command_embedding).item()
            if score > THRESHOLD:
                maybe_evict_oldest_speaker()
                speaker_db[user_id] = {
                    "embedding": embed,
                    "timestamp": time.time()
                }
                torch.save(speaker_db, SPEAKER_EMBEDDINGS_FILE)
                monitor_db_size()
                print(f"Re-inserted {user_id} into speaker_db.")
                return True
    return False

def verify_speaker():
    command_file = "command.wav"
    record_audio(command_file)
    
    sample_rate, audio_data = wav.read(command_file)
    if detect_watermark(audio_data, sample_rate):
        print("Warning: Watermark detected. Possible replay attack!")
        return False
    
    command_embedding = get_embedding(command_file)
    for user_id, data in speaker_db.items():
        score = verification.similarity(data["embedding"], command_embedding).item()
        print(f"User_id {user_id} Verification score: {score}")
        if score > THRESHOLD:
            print(f"Authentication successful for user_id {user_id} with score: {score}")
            data["timestamp"] = time.time()
            torch.save(speaker_db, SPEAKER_EMBEDDINGS_FILE)
            return True
    print("Not found in speaker_db. Checking uncached files...")
    if check_uncached_files(command_embedding):
        return True
    print("Authentication failed. No matching user found.")
    return False

def enroll_speaker(num):
    filename = f"enrolled_user_{num}.wav"
    print("Enrolling user ...")

    record_audio(filename, with_watermark=True)

    sample_rate, audio_data = wav.read(filename)
    
    sos = signal.butter(10, HIGH_FREQ_MIN - 500, btype='lowpass', fs=sample_rate, output='sos')
    filtered_audio = signal.sosfilt(sos, audio_data).astype(np.int16)
    
    filtered_filename = f"enrolled_user_{num}_filtered.wav"
    wav.write(filtered_filename, sample_rate, filtered_audio)
    
    embedding = get_embedding(filtered_filename)
    speaker_db[f"user_{num}"] = {
        "embedding": embedding,
        "timestamp": time.time()
    }

    torch.save(speaker_db, SPEAKER_EMBEDDINGS_FILE)
    monitor_db_size()
    maybe_evict_oldest_speaker()
    print("Enrollment complete.")
    
    os.remove(filtered_filename)

def get_next_user_index():
    max_index = -1
    for key in speaker_db:
        if key.startswith("user_"):
            idx = int(key.split('_')[1])
            if idx > max_index:
                max_index = idx
    return max_index + 1

def main():
    num_users = get_next_user_index()
    if num_users == 0:
        print("No users enrolled. Enroll a user first.")
        enroll_speaker(num_users)
        num_users += 1
    else:
        print(f"Currently enrolled users: {num_users}")

    active = input("Would you like to continue? y/n: ")

    while (active != "n"):
        enroll_new = input('Would you like to enroll a new user? y/n')

        if enroll_new == 'y':
            num_users = get_next_user_index()
            enroll_speaker(num_users)
            continue
        elif enroll_new == 'n':
            pass
        matched = verify_speaker()
        
        if matched:
            print("Command Accepted")
            if enroll_new == 'y':
                enroll_speaker(num_users)
                num_users+=1
            else:
                pass
                # use whisper here
        else:
            print("Command Denied")
            break
        
        active = input("Would you like to continue? Answer with yes or no.")

if __name__ == "__main__":
    main()
