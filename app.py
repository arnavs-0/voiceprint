from flask import Flask, render_template, request, jsonify
import os
import base64
import model
import time
from pydub import AudioSegment
import io

def convert_audio_data(audio_bytes, output_file, target_sample_rate=16000):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_frame_rate(target_sample_rate)
    
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Export as WAV file with correct parameters
    audio.export(output_file, format="wav", parameters=[
        "-ac", "1",
        "-ar", str(target_sample_rate),
        "-sample_fmt", "s16",
    ])
    
    return output_file

app = Flask(__name__)

@app.route('/')
def index():
    cache_users = len(model.speaker_db)
    cloud_users = 0
    enrolled_wavs = [f for f in os.listdir('.') if f.startswith("enrolled_user_") and f.endswith(".wav")]
    
    for wav_file in enrolled_wavs:
        file_name_parts = wav_file.replace('.wav', '').split('_')
        if len(file_name_parts) >= 3:
            user_id = '_'.join(file_name_parts[2:])
            found_in_cache = False
            
            for cache_user_id in model.speaker_db:
                if user_id in cache_user_id or cache_user_id in user_id:
                    found_in_cache = True
                    break
            
            if not found_in_cache:
                cloud_users += 1
    
    total_users = cache_users + cloud_users
    
    return render_template('index.html', num_users=total_users, 
                          cache_users=cache_users,
                          cloud_users=cloud_users)

@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    if request.method == 'GET':
        return render_template('enroll.html')
    
    elif request.method == 'POST':
        audio_data = request.form.get('audio')
        name = request.form.get('name')
        
        if not name:
            return jsonify({"success": False, "message": "Name is required"})
            
        if audio_data:
            try:
                audio_bytes = base64.b64decode(audio_data.split(',')[1])
                
                timestamp = int(time.time())
                user_id = f"{name.strip().replace(' ', '_')}_{timestamp}"
                filename = f"enrolled_user_{name.strip().replace(' ', '_')}_{timestamp}.wav"
                
                for existing_id in list(model.speaker_db.keys()):
                    if name.lower() in existing_id.lower():
                        return jsonify({
                            "success": False, 
                            "message": f"A user '{name}' is already enrolled. Please use a different name."
                        })
                
                convert_audio_data(audio_bytes, filename, target_sample_rate=model.SAMPLE_RATE)
                
                embedding = model.get_embedding(filename)
                model.speaker_db[user_id] = {
                    "embedding": embedding,
                    "timestamp": timestamp,
                    "display_name": name,
                }
                model.torch.save(model.speaker_db, model.SPEAKER_EMBEDDINGS_FILE)
                model.monitor_db_size()
                model.maybe_evict_oldest_speaker()
                
                return jsonify({"success": True, "message": f"User {name} enrolled successfully!"})
            except Exception as e:
                print(f"Error processing audio: {str(e)}")
                return jsonify({"success": False, "message": f"Error processing audio: {str(e)}"})
        
        return jsonify({"success": False, "message": "No audio data received"})

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'GET':
        return render_template('verify.html')
    
    elif request.method == 'POST':
        audio_data = request.form.get('audio')
        if audio_data:
            try:
                audio_bytes = base64.b64decode(audio_data.split(',')[1])
                command_file = "command.wav"
                
                convert_audio_data(audio_bytes, command_file, target_sample_rate=model.SAMPLE_RATE)
                
                sample_rate, audio = model.wav.read(command_file)
                watermark_detected = bool(model.detect_watermark(audio, sample_rate))
                
                command_embedding = model.get_embedding(command_file)
                authenticated = False
                matched_user = None
                display_name = None
                auth_score = 0.0
                
                if not watermark_detected:
                    for user_id, data in model.speaker_db.items():
                        score = model.verification.similarity(data["embedding"], command_embedding).item()
                        print(f"User {user_id} score: {score}")
                        if score > model.THRESHOLD:
                            authenticated = True
                            matched_user = user_id
                            display_name = data.get("display_name", user_id)
                            auth_score = float(score)
                            data["timestamp"] = time.time()
                            model.torch.save(model.speaker_db, model.SPEAKER_EMBEDDINGS_FILE)
                            break
                    
                    if not authenticated:
                        authenticated = model.check_uncached_files(command_embedding)
                
                spoofing_status = {
                    "passed_liveness": not watermark_detected,
                    "passed_realness": True
                }
                
                return jsonify({
                    "success": True, 
                    "authenticated": bool(authenticated) and not watermark_detected,  # Ensure boolean is Python native
                    "user": display_name if authenticated and not watermark_detected else None,
                    "user_id": matched_user if authenticated and not watermark_detected else None,
                    "auth_score": float(auth_score),  # Ensure score is a Python float
                    "spoofing_status": spoofing_status,
                    "watermark_detected": watermark_detected
                })
            except Exception as e:
                print(f"Error processing audio: {str(e)}")
                return jsonify({"success": False, "message": f"Error processing audio: {str(e)}"})
        
        return jsonify({"success": False, "message": "No audio data received"})

@app.route('/users')
def users():
    cached_users = []
    for user_id in model.speaker_db:
        timestamp = model.speaker_db[user_id]["timestamp"]
        display_name = model.speaker_db[user_id].get("display_name", user_id)
        
        embedding = model.speaker_db[user_id]["embedding"]
        embedding_size = embedding.element_size() * embedding.nelement()
        embedding_size_formatted = format_size(embedding_size)
        
        wav_size = 0
        wav_size_formatted = "N/A"
        wav_file = None
        
        if '_' in user_id:
            try:
                name_part = user_id.rsplit('_', 1)[0]
                timestamp_part = user_id.rsplit('_', 1)[1]
                filename_pattern = f"enrolled_user_{name_part}_{timestamp_part}"
            except:
                filename_pattern = f"enrolled_user_{user_id}"
        else:
            filename_pattern = f"enrolled_user_{user_id}"
        
        for file_name in os.listdir('.'):
            if file_name.startswith(filename_pattern) and file_name.endswith('.wav'):
                wav_file = file_name
                wav_size = os.path.getsize(file_name)
                wav_size_formatted = format_size(wav_size)
                break
        
        cached_users.append({
            "id": user_id,
            "name": display_name,
            "last_used": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
            "storage_type": "cache",
            "embedding_size": embedding_size_formatted,
            "wav_size": wav_size_formatted,
            "wav_file": wav_file
        })
    
    enrolled_wavs = [f for f in os.listdir('.') if f.startswith("enrolled_user_") and f.endswith(".wav")]
    cloud_users = []
    
    for wav_file in enrolled_wavs:
        file_name_parts = wav_file.replace('.wav', '').split('_')
        if len(file_name_parts) >= 3:
            try:
                if file_name_parts[-1].isdigit():
                    timestamp = int(file_name_parts[-1])
                else:
                    timestamp = os.path.getctime(wav_file)
                
                user_id = '_'.join(file_name_parts[2:])
                found_in_cache = False
                
                for cached_user in cached_users:
                    if user_id in cached_user["id"] or cached_user["id"] in user_id:
                        found_in_cache = True
                        break
                
                if not found_in_cache:
                    display_name = ' '.join(file_name_parts[2:-1]) if len(file_name_parts) > 3 else file_name_parts[2]
                    
                    wav_size = os.path.getsize(wav_file)
                    wav_size_formatted = format_size(wav_size)
                    
                    cloud_users.append({
                        "id": user_id,
                        "name": display_name,
                        "last_used": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
                        "storage_type": "cloud",
                        "embedding_size": "N/A",
                        "wav_size": wav_size_formatted,
                        "wav_file": wav_file
                    })
            except Exception as e:
                print(f"Error processing WAV file {wav_file}: {e}")
    
    users_list = cached_users + cloud_users
    users_list.sort(key=lambda x: x["last_used"], reverse=True)
    
    return render_template('users.html', users=users_list)

def format_size(size_bytes):
    """Format file size in a human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

@app.route('/delete_user', methods=['POST'])
def delete_user():
    user_id = request.form.get('user_id')
    wav_file = request.form.get('wav_file')
    
    if user_id:
        deleted_files = []
        
        if '_' in user_id:
            try:
                name_part = user_id.rsplit('_', 1)[0]
                timestamp_part = user_id.rsplit('_', 1)[1] 
                filename_pattern = f"enrolled_user_{name_part}_{timestamp_part}"
            except:
                filename_pattern = f"enrolled_user_{user_id}"
        else:
            filename_pattern = f"enrolled_user_{user_id}"
        
        for file_name in os.listdir('.'):
            if file_name.startswith(filename_pattern) and file_name.endswith('.wav'):
                try:
                    os.remove(file_name)
                    deleted_files.append(file_name)
                    print(f"Deleted file: {file_name}")
                except Exception as e:
                    print(f"Error deleting file {file_name}: {e}")
        
        if wav_file and os.path.exists(wav_file):
            try:
                os.remove(wav_file)
                deleted_files.append(wav_file)
                print(f"Deleted specified WAV file: {wav_file}")
            except Exception as e:
                print(f"Error deleting specified WAV file {wav_file}: {e}")
        
        if user_id in model.speaker_db:
            del model.speaker_db[user_id]
            model.torch.save(model.speaker_db, model.SPEAKER_EMBEDDINGS_FILE)
            print(f"Removed user {user_id} from speaker database")
        
        if deleted_files or user_id in model.speaker_db:
            return jsonify({
                "success": True, 
                "message": f"User deleted successfully. Files removed: {', '.join(deleted_files) if deleted_files else 'None'}"
            })
    
    return jsonify({"success": False, "message": "User not found or nothing to delete"})

if __name__ == '__main__':
    app.run(debug=True)
