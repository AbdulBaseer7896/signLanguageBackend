import cv2
import mediapipe as mp
import time
import threading
import numpy as np
from collections import deque, defaultdict
from tensorflow.keras.models import load_model
import queue

print("Test 1")
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit(1)  #
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
print("Test 2")
capture_times = deque(maxlen=30)
process_times = deque(maxlen=30)
processed_frame_rate = 15
frame_skip_interval = max(1, int(30 / processed_frame_rate))
data_list = np.zeros(135, dtype=np.float32)
model = load_model('dense_97_200_64_218_words_new.h5', compile=False)
words = ['a', 'able', 'absorb', 'afraid', 'airplane', 'almost', 'also', 'amazing', 'ancient', 'angry', 'announce', 'appreciate', 'ashamed', 'ask', 'attack', 'awake', 'b', 'bad', 'balcony', 'beg', 'bend', 'big', 'blank', 'blender', 'blink', 'blow', 'brake', 'bring', 'c', 'calm', 'care', 'carry', 'clever', 'cold', 'comb', 'come', 'complain', 'conceive', 'concentrate', 'confess', 'consider', 'continue', 'coordinate', 'copy', 'cost', 'crazy', 'crooked', 'cry', 'cure', 'cute', 'd', 'daily', 'damn', 'dangerous', 'deep', 'difficult', 'dip', 'dirty', 'distract', 'dizzy', 'double', 'doubtful', 'dream', 'drink', 'dumb', 'e', 'early', 'eat', 'empty', 'energetic', 'every', 'f', 'famous', 'fan', 'fantastic', 'fast', 'female', 'find', 'foolish', 'future', 'g', 'general', 'generous', 'gentle', 'get', 'give', 'glad', 'go', 'good', 'greedy', 'grow', 'h', 'happen', 'hate', 'hear', 'hence', 'honest', 'hungry', 'i', 'impatient', 'impossible', 'impress', 'incorrect', 'increase', 'inferior', 'insist', 'insult', 'intelligent', 'internal', 'inward', 'isolate', 'j', 'jealous', 'k', 'key', 'kiss', 'knock', 'know', 'l', 'left_side', 'lick', 'lock', 'look', 'm', 'mad', 'male', 'material', 'mild', 'military', 'miserable', 'modern', 'monthly', 'n', 'natural', 'negative', 'neglect', 'neutral', 'new', 'nice', 'normal', 'not', 'now', 'o', 'observe', 'occasionally', 'old', 'only', 'order', 'original', 'outdoors', 'p', 'particular', 'pass', 'past', 'perfume', 'picture', 'positive', 'possible', 'proud', 'q', 'r', 'razor', 'refuse', 'regular', 'repeat', 'reserve', 'respect', 'return', 'right_side', 'rigid', 'roof', 'round', 's', 'sad', 'salute', 'save', 'say', 'search', 'second', 'sell', 'send', 'shiny', 'shoot', 'short_height', 'shower', 'shy', 'silly', 'sink', 'slow', 'small', 'smile', 'sniff', 'speak', 'stairs', 'stare', 'sticky', 'stiff', 'stir', 'stop', 'straight', 'strong', 'subtle', 'swallow', 't', 'telephone', 'tell', 'thick', 'thin', 'toothbrush', 'u', 'v', 'w', 'wall', 'weak', 'win', 'x', 'y', 'z']
print("Test 3")
frame_queue = queue.Queue(maxsize=5)
data_queue = queue.Queue(maxsize=10)
prediction_queue = queue.Queue(maxsize=10)
stop_event = threading.Event()
print("Test 4")
def process_pose(frame_rgb, pose_instance):
    try:
        return pose_instance.process(frame_rgb)
    except Exception as e:
        print(f"Pose processing error: {e}")
        return None

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

def pose_hand_worker(frame_queue, data_queue, stop_event):
    pose_instance = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=0, static_image_mode=False)
    hands_instance = mp_hands.Hands(max_num_hands=2, model_complexity=0, min_detection_confidence=0.3, min_tracking_confidence=0.3, static_image_mode=False)
    while not stop_event.is_set():
        try:
            frame_rgb, frame_time = frame_queue.get(timeout=1.0)
            pose_results = process_pose(frame_rgb, pose_instance)
            hand_results = process_hands(frame_rgb, hands_instance)
            results, hand_detected = process_and_print_results(pose_results, hand_results)
            data_queue.put((results, hand_detected, pose_results, frame_time))
            frame_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Pose/hand worker error: {e}")
    pose_instance.close()
    hands_instance.close()

def prediction_worker(data_queue, prediction_queue, stop_event):
    while not stop_event.is_set():
        try:
            results, hand_detected, pose_results, frame_time = data_queue.get(timeout=1.0)
            if hand_detected:
                results_reshaped = np.array([results])
                prediction_proba = model.predict(results_reshaped, verbose=0)[0]
                prediction_index = np.argmax(prediction_proba)
                prediction_confidence = float(prediction_proba[prediction_index] * 100)
                prediction_queue.put((words[prediction_index], prediction_confidence, frame_time, hand_detected))
            data_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Prediction worker error: {e}")

pose_hand_thread = threading.Thread(target=pose_hand_worker, args=(frame_queue, data_queue, stop_event))
prediction_thread = threading.Thread(target=prediction_worker, args=(data_queue, prediction_queue, stop_event))
pose_hand_thread.start()
prediction_thread.start()

frame_counter = 0
last_fps_update = time.time()
last_prediction_print = time.time()
current_capture_fps = 0.0
current_process_fps = 0.0
initial_flip = True
prediction_history = []
predicted_word = ""
current_scale_factor = 1.0
target_scale_factor = 1.0
scale_step = 0.02
last_distance_check = time.time()
distance_check_interval = 0.5
distance_threshold = 1.3
min_distance_threshold = 0.8
ideal_max_distance = 1.0
user_present_counter = 0
head_position_alert = None
last_head_check = time.time()
head_check_interval = 1.0
is_prediction_paused = False
distance_alert = None
is_hand_detected = False
previous_hand_detected_state = False

# New variables for improved timing control
collection_start_time = None
collection_duration = 1.5  # Duration for collecting predictions (1.5 seconds)
display_start_time = None
display_duration = 1.5  # Duration to display the predicted word (1.5 seconds)
is_collecting = False
is_displaying = False
min_predictions_to_process = 3
print("Test 5")
main_pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=0, static_image_mode=False)
print("Test 6")
try:
    while cap.isOpened():
        print("Test 7")
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            time.sleep(0.1)
            continue
        
        current_time = time.time()
        capture_times.append(current_time)
        frame_height, frame_width = frame.shape[:2]
        
        if initial_flip:
            frame = cv2.flip(frame, 1)
        
        is_too_close = False
        is_too_far = False
        
        if frame_counter % (frame_skip_interval * 2) == 0:
            low_res_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            low_res_pose = process_pose(low_res_rgb, main_pose)
            
            user_detected = low_res_pose and low_res_pose.pose_landmarks and any(lm.visibility > 0.5 for lm in low_res_pose.pose_landmarks.landmark[:5])
            
            if not user_detected:
                user_present_counter += 1
                if user_present_counter > 15:
                    cv2.putText(frame, "No user detected", (frame_width // 2 - 100, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                user_present_counter = 0
                if current_time - last_head_check >= head_check_interval:
                    head_positioned_well, position_message = check_head_position(low_res_pose, frame_height)
                    head_position_alert = position_message if not head_positioned_well else None
                    last_head_check = current_time
                
                if current_time - last_distance_check >= distance_check_interval:
                    distance = calculate_distance(low_res_pose, frame_width)
                    if distance:
                        if distance < min_distance_threshold or distance <= ideal_max_distance:
                            target_scale_factor = 1.0
                            is_too_close = True
                            distance_alert = "Please move back for better detection"
                        elif distance > distance_threshold:
                            target_scale_factor = min(1.0 + (distance - distance_threshold) * 0.75, 2.5)
                            is_too_far = True
                            distance_alert = None
                        else:
                            target_scale_factor = 1.0
                            distance_alert = None
                    last_distance_check = current_time
        
        if abs(current_scale_factor - target_scale_factor) > scale_step:
            direction = 1 if current_scale_factor < target_scale_factor else -1
            current_scale_factor += direction * scale_step
        else:
            current_scale_factor = target_scale_factor
        
        display_frame = smooth_zoom(frame, current_scale_factor, frame_width, frame_height)
        
        if frame_counter % frame_skip_interval == 0:
            process_start_time = time.time()
            processing_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                if not frame_queue.full():
                    frame_queue.put((processing_frame_rgb, process_start_time))
                process_times.append(time.time())
            except Exception as e:
                print(f"Queue error: {e}")
        
        is_prediction_paused = head_position_alert is not None or is_too_close
        
        previous_hand_detected_state = is_hand_detected
        
        # Process predictions from queue
        while not prediction_queue.empty():
            try:
                word, confidence, frame_time, hand_detected = prediction_queue.get_nowait()
                is_hand_detected = hand_detected
                
                if not is_prediction_paused and is_hand_detected and is_collecting:
                    prediction_history.append((word, confidence))
                
                prediction_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error processing prediction queue: {e}")
        
        # Hand state transition and collection timing logic
        if is_hand_detected and not is_prediction_paused:
            # If hand is detected, we should be either collecting or displaying
            if not is_collecting and not is_displaying:
                # Start a new collection period
                collection_start_time = current_time
                is_collecting = True
                prediction_history = []
                print(f"Starting new collection at {current_time}")
            
            # Check if we need to end collection and process results
            if is_collecting and current_time - collection_start_time >= collection_duration:
                # End collection and process
                if len(prediction_history) >= min_predictions_to_process:
                    print(f"Processing batch with {len(prediction_history)} predictions after collection period")
                    predicted_word = process_prediction_batch(prediction_history)
                    print(f"Selected prediction: {predicted_word}")
                    is_displaying = True
                    display_start_time = current_time
                else:
                    print(f"Discarding batch with insufficient predictions: {len(prediction_history)}")
                
                # Reset collection state
                is_collecting = False
                prediction_history = []
        
        # If we lost hand detection during collection, wait for hand to reappear
        elif not is_hand_detected and is_collecting:
            # Hand disappeared during collection, reset collection but don't start display yet
            print("Hand lost during collection, resetting")
            is_collecting = False
            prediction_history = []
        
        # Check if display period has expired
        if is_displaying and current_time - display_start_time >= display_duration:
            is_displaying = False
            # If hand is still detected, start collecting again
            if is_hand_detected and not is_prediction_paused:
                collection_start_time = current_time
                is_collecting = True
                prediction_history = []
                print(f"Starting new collection at {current_time} after display period")
        
        if current_time - last_fps_update >= 0.5:
            current_capture_fps = calculate_fps(capture_times)
            current_process_fps = calculate_fps(process_times)
            last_fps_update = current_time
        
        # Debug print current predictions
        if current_time - last_prediction_print >= 1.0 and prediction_history and is_collecting:
            print(f"Current predictions: {prediction_history}")
            last_prediction_print = current_time
        
        # Display info on screen
        if current_scale_factor > 1.0:
            zoom_percentage = int((current_scale_factor - 1.0) * 100)
            cv2.putText(display_frame, f"Zoom: {zoom_percentage}%", (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        cv2.putText(display_frame, f"Capture FPS: {current_capture_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Process FPS: {current_process_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display prediction status
        if is_prediction_paused:
            cv2.putText(display_frame, "Prediction Paused", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif not is_hand_detected:
            cv2.putText(display_frame, "No Hand Detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif is_collecting:
            remaining = max(0, collection_duration - (current_time - collection_start_time))
            cv2.putText(display_frame, f"Collecting: {remaining:.1f}s ({len(prediction_history)})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif is_displaying and predicted_word:
            remaining_display = max(0, display_duration - (current_time - display_start_time))
            cv2.putText(display_frame, f"Prediction: {predicted_word} ({remaining_display:.1f}s)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw prediction with bigger font in center of screen
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(predicted_word, font, 1.5, 3)[0]
            text_x = (frame_width - text_size[0]) // 2
            text_y = (frame_height + text_size[1]) // 2
            
            # Draw background for better visibility
            cv2.rectangle(display_frame, 
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)
            cv2.putText(display_frame, predicted_word, (text_x, text_y), font, 1.5, (0, 255, 0), 3)
        
        if head_position_alert:
            display_frame = draw_alert_box(display_frame, f"Please {head_position_alert} for better detection", "warning")
        
        if distance_alert and current_scale_factor <= 1.0:
            display_frame = draw_alert_box(display_frame, distance_alert, "warning")
        
        cv2.imshow('MediaPipe Pose and Hands', display_frame)
        frame_counter += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break
except Exception as e:
    print(f"Main loop error: {e}")
finally:
    stop_event.set()
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
            frame_queue.task_done()
        except queue.Empty:
            break
    while not data_queue.empty():
        try:
            data_queue.get_nowait()
            data_queue.task_done()
        except queue.Empty:
            break
    while not prediction_queue.empty():
        try:
            prediction_queue.get_nowait()
            prediction_queue.task_done()
        except queue.Empty:
            break
    pose_hand_thread.join()
    prediction_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    main_pose.close()



print("Test 8")