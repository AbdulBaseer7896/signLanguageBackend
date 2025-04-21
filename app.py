from flask import Flask, Response, render_template_string
import cv2
import mediapipe as mp
import threading
import numpy as np
from collections import deque, defaultdict
from tensorflow.keras.models import load_model
import queue
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)
latest_prediction = ""
latest_frame = None
frame_lock = threading.Lock()


# Initialize MediaPipe and model components
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Model and word list setup
model = load_model('dense_97_200_64_218_words_new.h5', compile=False)
words = ['a', 'able', 'absorb', 'afraid', 'airplane', 'almost', 'also', 'amazing', 'ancient', 'angry', 'announce', 'appreciate', 'ashamed', 'ask', 'attack', 'awake', 'b', 'bad', 'balcony', 'beg', 'bend', 'big', 'blank', 'blender', 'blink', 'blow', 'brake', 'bring', 'c', 'calm', 'care', 'carry', 'clever', 'cold', 'comb', 'come', 'complain', 'conceive', 'concentrate', 'confess', 'consider', 'continue', 'coordinate', 'copy', 'cost', 'crazy', 'crooked', 'cry', 'cure', 'cute', 'd', 'daily', 'damn', 'dangerous', 'deep', 'difficult', 'dip', 'dirty', 'distract', 'dizzy', 'double', 'doubtful', 'dream', 'drink', 'dumb', 'e', 'early', 'eat', 'empty', 'energetic', 'every', 'f', 'famous', 'fan', 'fantastic', 'fast', 'female', 'find', 'foolish', 'future', 'g', 'general', 'generous', 'gentle', 'get', 'give', 'glad', 'go', 'good', 'greedy', 'grow', 'h', 'happen', 'hate', 'hear', 'hence', 'honest', 'hungry', 'i', 'impatient', 'impossible', 'impress', 'incorrect', 'increase', 'inferior', 'insist', 'insult', 'intelligent', 'internal', 'inward', 'isolate', 'j', 'jealous', 'k', 'key', 'kiss', 'knock', 'know', 'l', 'left_side', 'lick', 'lock', 'look', 'm', 'mad', 'male', 'material', 'mild', 'military', 'miserable', 'modern', 'monthly', 'n', 'natural', 'negative', 'neglect', 'neutral', 'new', 'nice', 'normal', 'not', 'now', 'o', 'observe', 'occasionally', 'old', 'only', 'order', 'original', 'outdoors', 'p', 'particular', 'pass', 'past', 'perfume', 'picture', 'positive', 'possible', 'proud', 'q', 'r', 'razor', 'refuse', 'regular', 'repeat', 'reserve', 'respect', 'return', 'right_side', 'rigid', 'roof', 'round', 's', 'sad', 'salute', 'save', 'say', 'search', 'second', 'sell', 'send', 'shiny', 'shoot', 'short_height', 'shower', 'shy', 'silly', 'sink', 'slow', 'small', 'smile', 'sniff', 'speak', 'stairs', 'stare', 'sticky', 'stiff', 'stir', 'stop', 'straight', 'strong', 'subtle', 'swallow', 't', 'telephone', 'tell', 'thick', 'thin', 'toothbrush', 'u', 'v', 'w', 'wall', 'weak', 'win', 'x', 'y', 'z']
# Keep your full word list

# Threading and queue setup
frame_queue = queue.Queue(maxsize=5)
data_queue = queue.Queue(maxsize=10)
prediction_queue = queue.Queue(maxsize=10)
stop_event = threading.Event()

# Global variables for sharing data between threads
latest_prediction = ""
latest_frame = None
frame_lock = threading.Lock()
initial_frame = np.zeros((480, 640, 3), dtype=np.uint8)
# HTML Template
# Backend Modifications (Flask)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Sign Language Recognition</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            text-align: center;
        }

        .control-buttons {
            margin: 2rem 0;
            display: flex;
            gap: 1rem;
            justify-content: center;
        }

        .button {
            padding: 1rem 2rem;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .start-button {
            background: #00c853;
            color: white;
            box-shadow: 0 4px 15px rgba(0,200,83,0.4);
        }

        .stop-button {
            background: #ff1744;
            color: white;
            box-shadow: 0 4px 15px rgba(255,23,68,0.4);
            display: none;
        }

        .button:hover {
            transform: translateY(-2px);
            opacity: 0.9;
        }

        .video-container {
            margin: 2rem auto;
            width: 640px;
            position: relative;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            display: none;
        }

        #videoFeed {
            width: 100%;
            height: auto;
            background: #000;
        }

        .prediction-box {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.7);
            padding: 1rem 2rem;
            border-radius: 30px;
            font-size: 1.5rem;
            backdrop-filter: blur(5px);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sign Language Interpreter</h1>
        
        <div class="control-buttons">
            <button class="button start-button" onclick="startCamera()">
                <span class="material-icons">play_arrow</span>
                Start Camera
            </button>
            <button class="button stop-button" onclick="stopCamera()">
                <span class="material-icons">stop</span>
                Stop Camera
            </button>
        </div>

        <div class="video-container" id="videoContainer">
            <img id="videoFeed" src="">
            <div class="prediction-box" id="prediction">Ready</div>
        </div>
    </div>

    <script>
        let predictionInterval;
        const videoContainer = document.getElementById('videoContainer');
        const startBtn = document.querySelector('.start-button');
        const stopBtn = document.querySelector('.stop-button');

        function startCamera() {
            // Toggle buttons
            startBtn.style.display = 'none';
            stopBtn.style.display = 'flex';
            videoContainer.style.display = 'block';

            // Initialize video feed
            const videoFeed = document.getElementById('videoFeed');
            videoFeed.src = 'http://localhost:5000/video_feed?' + Date.now();

            // Start prediction updates
            predictionInterval = setInterval(updatePrediction, 500);
            
            // Initialize camera in backend
            fetch('/start_camera');
        }

        function stopCamera() {
            // Toggle buttons
            startBtn.style.display = 'flex';
            stopBtn.style.display = 'none';
            videoContainer.style.display = 'none';

            // Stop prediction updates
            clearInterval(predictionInterval);
            document.getElementById('prediction').textContent = 'Stopped';
            
            // Clear video feed
            const videoFeed = document.getElementById('videoFeed');
            videoFeed.src = '';
            
            // Stop camera in backend
            fetch('/stop_camera');
        }

        function updatePrediction() {
            fetch('/prediction')
                .then(response => response.text())
                .then(data => {
                    document.getElementById('prediction').textContent = data || 'No prediction';
                });
        }
    </script>
</body>
</html>
"""

# Add these routes to your Flask app
@app.route('/start_camera')
def start_camera():
    global processing_thread, prediction_thread, stop_event
    
    if not processing_thread.is_alive():
        stop_event.clear()
        processing_thread = threading.Thread(target=pose_hand_worker)
        prediction_thread = threading.Thread(target=prediction_worker)
        processing_thread.start()
        prediction_thread.start()
    
    return '', 204

@app.route('/stop_camera')
def stop_camera():
    global stop_event
    stop_event.set()
    
    # Release camera resources
    with frame_lock:
        global latest_frame
        latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    return '', 204

def generate_frames():
    while True:
        with frame_lock:
            current_frame = latest_frame if latest_frame is not None else initial_frame
            ret, buffer = cv2.imencode('.jpg', current_frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML ,  prediction=latest_prediction)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def get_prediction():
    return latest_prediction
def process_pose(frame_rgb, pose_instance):
    try:
        return pose_instance.process(frame_rgb)
    except Exception as e:
        print(f"Pose processing error: {e}")
        return None

# Include all your processing functions (process_hands, process_single_hand, 
# process_and_print_results, calculate_fps, etc.) here...


def process_hands(frame_rgb, hands_instance):
    try:
        return hands_instance.process(frame_rgb)
    except Exception as e:
        print(f"Hand processing error: {e}")
        return None

def process_single_hand(results_hand, combined_data):
    if not results_hand.multi_hand_landmarks:
        combined_data[:126] = 0
        return combined_data, False
    left_found = right_found = False
    for hand_landmarks in results_hand.multi_hand_landmarks:
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32).flatten()
        if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
            combined_data[:63] = landmarks_array
            left_found = True
        else:
            combined_data[63:126] = landmarks_array
            right_found = True
    if not left_found:
        combined_data[:63] = 0
    if not right_found:
        combined_data[63:126] = 0
    return combined_data, True

def process_and_print_results(pose_results, hand_results):
    data_list = np.zeros(135, dtype=np.float32)
    hand_detected = False
    if hand_results:
        data_list, hand_detected = process_single_hand(hand_results, data_list)
    if pose_results and pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        for idx, pos in enumerate([0, 11, 12]):
            lm = landmarks[pos]
            base_idx = 126 + (idx * 3)
            data_list[base_idx:base_idx + 3] = [lm.x, lm.y, lm.z]
    return data_list, hand_detected

def calculate_fps(time_deque):
    if len(time_deque) < 2:
        return 0.0
    try:
        time_diff = time_deque[-1] - time_deque[0]
        if time_diff <= 0:
            return 0.0
        return len(time_deque) / time_diff
    except Exception:
        return 0.0

def calculate_distance(pose_results, frame_width):
    if not pose_results or not pose_results.pose_landmarks:
        return None
    landmarks = pose_results.pose_landmarks.landmark
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
        return None
    pixel_distance = np.sqrt((left_shoulder.x - right_shoulder.x)**2 * frame_width**2)
    reference_shoulder_width = 0.4
    focal_length = 640
    estimated_distance = (reference_shoulder_width * focal_length) / pixel_distance
    return estimated_distance

def check_head_position(pose_results, frame_height):
    if not pose_results or not pose_results.pose_landmarks:
        return True, None
    landmarks = pose_results.pose_landmarks.landmark
    nose = landmarks[0]
    if nose.visibility < 0.7:
        return True, None
    nose_y_pixel = int(nose.y * frame_height)
    ideal_top_margin = int(frame_height * 0.1)
    if nose_y_pixel < ideal_top_margin:
        return False, "Move down"
    elif nose_y_pixel > frame_height * 0.4:
        return False, "Move up"
    else:
        return True, None

def smooth_zoom(frame, current_scale, frame_width, frame_height):
    if current_scale <= 1.0:
        return frame
    new_width = int(frame_width / current_scale)
    new_height = int(frame_height / current_scale)
    if new_width <= 0 or new_height <= 0 or new_width > frame_width or new_height > frame_height:
        return frame
    x_start = max(0, (frame_width - new_width) // 2)
    y_start = max(0, (frame_height - new_height) // 2)
    cropped_frame = frame[y_start:y_start + new_height, x_start:x_start + new_width]
    zoomed_frame = cv2.resize(cropped_frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
    return zoomed_frame

def draw_alert_box(frame, message, severity="warning"):
    frame_height, frame_width = frame.shape[:2]
    color = (0, 165, 255) if severity == "warning" else (0, 0, 255) if severity == "error" else (0, 255, 0)
    overlay = frame.copy()
    box_height = 40
    box_width = int(frame_width * 0.6)
    start_x = (frame_width - box_width) // 2
    start_y = frame_height - box_height - 20
    cv2.rectangle(overlay, (start_x, start_y), (start_x + box_width, start_y + box_height), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 0.7, 2)[0]
    text_x = start_x + (box_width - text_size[0]) // 2
    text_y = start_y + (box_height + text_size[1]) // 2
    cv2.putText(frame, message, (text_x, text_y), font, 0.7, (255, 255, 255), 2)
    return frame

def process_prediction_batch(predictions):
    if not predictions:
        return ""
    
    freq = defaultdict(int)
    conf = defaultdict(list)
    
    for word, confidence in predictions:
        freq[word] += 1
        conf[word].append(confidence)
    
    total_predictions = len(predictions)
    
    if total_predictions <= 2:
        high_conf_candidates = [word for word in freq if any(c > 75 for c in conf[word])]
        if high_conf_candidates:
            if len(high_conf_candidates) == 1:
                return high_conf_candidates[0]
            else:
                avg_conf = {word: np.mean(conf[word]) for word in high_conf_candidates}
                return max(avg_conf, key=avg_conf.get)
        else:
            return ""
    else:
        max_freq = max(freq.values()) if freq else 0
        if max_freq <= 2:
            high_conf_candidates = [word for word in freq if any(c > 75 for c in conf[word])]
            if high_conf_candidates:
                if len(high_conf_candidates) == 1:
                    return high_conf_candidates[0]
                else:
                    avg_conf = {word: np.mean(conf[word]) for word in high_conf_candidates}
                    return max(avg_conf, key=avg_conf.get)
            else:
                return ""
        else:
            high_freq_candidates = [word for word, count in freq.items() if count == max_freq]
            valid_candidates = [word for word in high_freq_candidates if np.mean(conf[word]) >= 75]
            
            if valid_candidates:
                if len(valid_candidates) == 1:
                    return valid_candidates[0]
                else:
                    avg_conf = {word: np.mean(conf[word]) for word in valid_candidates}
                    return max(avg_conf, key=avg_conf.get)
            else:
                if total_predictions % 2 == 0:
                    threshold = (total_predictions // 2) - 1
                else:
                    threshold = total_predictions // 2
                
                threshold_candidates = [word for word, count in freq.items() if count >= threshold]
                
                if threshold_candidates:
                    max_threshold_freq = max(freq[word] for word in threshold_candidates)
                    max_freq_words = [word for word in threshold_candidates if freq[word] == max_threshold_freq]
                    
                    if len(max_freq_words) == 1:
                        return max_freq_words[0]
                    else:
                        avg_conf = {word: np.mean(conf[word]) for word in max_freq_words}
                        return max(avg_conf, key=avg_conf.get)
                elif total_predictions >= 4:
                    half_minus_one = (total_predictions // 2) - 1
                    fallback_candidates = [word for word, count in freq.items() if count >= half_minus_one]
                    if fallback_candidates:
                        max_fallback_freq = max(freq[word] for word in fallback_candidates)
                        max_freq_words = [word for word in fallback_candidates if freq[word] == max_fallback_freq]
                        if len(max_freq_words) == 1:
                            return max_freq_words[0]
                        else:
                            avg_conf = {word: np.mean(conf[word]) for word in max_freq_words}
                            return max(avg_conf, key=avg_conf.get)
                    else:
                        return ""
                else:
                    return ""

def pose_hand_worker():
    pose_instance = mp_pose.Pose(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=0,
        static_image_mode=False
    )
    hands_instance = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        static_image_mode=False
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize with black frame
    initial_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with frame_lock:
        global latest_frame
        latest_frame = initial_frame.copy()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        # Mirror and process frame
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process pose and hands
        pose_results = process_pose(frame_rgb, pose_instance)
        hand_results = process_hands(frame_rgb, hands_instance)

        # Get data array for model
        data_list, hand_detected = process_and_print_results(pose_results, hand_results)

        # Send to prediction queue if hands detected
        if hand_detected:
            try:
                data_queue.put(data_list, block=False)
            except queue.Full:
                pass

        # Add prediction text to frame
        frame_with_text = frame.copy()
        cv2.putText(frame_with_text, latest_prediction, 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 0), 2, cv2.LINE_AA)

        # Update shared frame with thread safety
        with frame_lock:
            latest_frame = frame_with_text

    # Cleanup
    cap.release()
    pose_instance.close()
    hands_instance.close()




def prediction_worker():
    batch_size = 5  # Adjust batch size as needed
    prediction_buffer = []
    while not stop_event.is_set():
        try:
            data = data_queue.get(timeout=1)
            prediction_buffer.append(data)
            if len(prediction_buffer) >= batch_size:
                predictions = []
                for data_point in prediction_buffer:
                    data_array = np.array([data_point], dtype=np.float32)
                    pred = model.predict(data_array, verbose=0)[0]
                    pred_index = np.argmax(pred)
                    confidence = np.max(pred) * 100
                    word = words[pred_index]
                    predictions.append((word, confidence))
                
                final_word = process_prediction_batch(predictions)
                global latest_prediction
                latest_prediction = final_word if final_word else latest_prediction
                print(f"Predictions: {predictions} → Final: {latest_prediction}")
                
                prediction_buffer.clear()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Prediction error: {e}")

if __name__ == '__main__':
    processing_thread = threading.Thread(target=pose_hand_worker)
    prediction_thread = threading.Thread(target=prediction_worker)
    processing_thread.start()
    prediction_thread.start()
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        stop_event.set()
        processing_thread.join()
        prediction_thread.join()