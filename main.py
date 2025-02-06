import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time

# ✅ Spotify API Credentials
SPOTIPY_CLIENT_ID = "982322e125b64067ac1f1f6bd7a27342"
SPOTIPY_CLIENT_SECRET = "5c6a3c13f210438d9264160c85ac7608"
SPOTIPY_REDIRECT_URI = "http://localhost:8888/callback"

# ✅ Authenticate Spotify
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-modify-playback-state,user-read-playback-state"
))

# ✅ Emotion to Playlist Mapping
emotion_to_playlist = {
    "happy": "https://open.spotify.com/playlist/35HdPwNtW6DqHydgcyLtSW?si=fdd321bdb3814c97",
    "sad": "https://open.spotify.com/playlist/1sxEZi9mKGu9Y2390IyfZW?si=f601c041366f4e3c",
    "angry": "https://open.spotify.com/playlist/37i9dQZF1EIgNZCaOGb0Mi?si=4cf9046ee5f842a6",
    "surprise": "https://open.spotify.com/playlist/6sz77RIbF2sMJhT75UYX2R?si=8799993727e94ea4"
}

# ✅ Start Webcam and Emotion Detection
cap = cv2.VideoCapture(0)
last_played_emotion = None  # Track last played emotion
last_play_time = 0  # Track last play timestamp


def get_active_device():
    """ Check for an active Spotify device """
    devices = sp.devices()
    
    if not devices["devices"]:
        print("⚠️ No active Spotify device found. Make sure Spotify is open and playing on a device.")
        return None
    
    # Look for an actively playing device
    for device in devices["devices"]:
        if device.get("is_active"):
            print(f"✅ Using active device: {device['name']}")
            return device["id"]
    
    # If no active device, return the first available one
    print(f"⚠️ No active device found. Using the first available device: {devices['devices'][0]['name']}")
    return devices["devices"][0]["id"]



def play_music(emotion):
    """ Play music for detected emotion if an active device is available """
    global last_played_emotion, last_play_time

    # Prevent frequent playback requests (5 seconds cooldown)
    if emotion == last_played_emotion and time.time() - last_play_time < 5:
        return

    playlist_uri = emotion_to_playlist.get(emotion.lower())
    if not playlist_uri:
        print(f"⚠️ No music found for emotion: {emotion}")
        return

    device_id = get_active_device()
    if device_id:
        try:
            # Pause briefly to allow device selection
            time.sleep(1)
            sp.start_playback(device_id=device_id, context_uri=playlist_uri)
            print(f"✅ Playing {emotion} music on {device_id}!")
            last_played_emotion = emotion
            last_play_time = time.time()
        except spotipy.exceptions.SpotifyException as e:
            print(f"🚨 Spotify Playback Error: {e}")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']
        print(f"🎭 Detected Emotion: {dominant_emotion}")

        # Play music based on detected emotion
        if dominant_emotion:
            play_music(dominant_emotion)

    except Exception as e:
        print(f"⚠️ Error in face analysis: {e}")

    # Display emotion label on video feed
    cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
