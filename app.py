# app.py
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Spotify API Credentials
SPOTIPY_CLIENT_ID = "982322e125b64067ac1f1f6bd7a27342"
SPOTIPY_CLIENT_SECRET = "5c6a3c13f210438d9264160c85ac7608"
SPOTIPY_REDIRECT_URI = "http://localhost:8888/callback"

# Initialize Spotify
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-modify-playback-state,user-read-playback-state"
))

emotion_to_playlist = {
    "happy": "https://open.spotify.com/playlist/35HdPwNtW6DqHydgcyLtSW?si=fdd321bdb3814c97",
    "sad": "https://open.spotify.com/playlist/1sxEZi9mKGu9Y2390IyfZW?si=f601c041366f4e3c",
    "angry": "https://open.spotify.com/playlist/37i9dQZF1EIgNZCaOGb0Mi?si=4cf9046ee5f842a6",
    "surprise": "https://open.spotify.com/playlist/6sz77RIbF2sMJhT75UYX2R?si=8799993727e94ea4"
}

camera = None
last_played_emotion = None
last_play_time = 0

def get_active_device():
    devices = sp.devices()
    if not devices["devices"]:
        return None
    for device in devices["devices"]:
        if device.get("is_active"):
            return device["id"]
    return devices["devices"][0]["id"] if devices["devices"] else None

def play_music(emotion):
    global last_played_emotion, last_play_time
    if emotion == last_played_emotion and time.time() - last_play_time < 5:
        return

    playlist_uri = emotion_to_playlist.get(emotion.lower())
    if not playlist_uri:
        return

    device_id = get_active_device()
    if device_id:
        try:
            time.sleep(1)
            sp.start_playback(device_id=device_id, context_uri=playlist_uri)
            last_played_emotion = emotion
            last_play_time = time.time()
            return True
        except spotipy.exceptions.SpotifyException:
            return False
    return False

def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']
            
            # Play music based on emotion
            play_music(dominant_emotion)
            
            # Add emotion text to frame
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error in face analysis: {e}")
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)