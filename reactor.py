import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles # Added for drawing

# Configuration (tweak thresholds below if needed)
SMILE_THRESHOLD = 0.32
OPEN_MOUTH_THRESHOLD = 0.50  # if MAR > this, mouth is considered open (tongue likely if red detected)
TONGUE_R_RATIO = 1.2  # mean R must be this times mean G to consider tongue (heuristic)
TONGUE_MIN_R = 80
TONGUE_RED_FRAC = 0.06  # fraction of mouth ROI pixels that must be red-ish to count as tongue
AXE_MAX_X_DIFF = 0.12  # wrists must be close together (normalized coords)
AXE_SIDE_X = 0.5  # both wrists must be on left side (x < 0.5)
RAISED_HAND_OFFSET = 0.03  # how much above shoulder y to consider as raised
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# --- DEBUGGING: INCREASED ANGRY THRESHOLD ---
# 0.01 was likely too small and hard to trigger. Let's try 0.025
ANGRY_THRESHOLD = 0.025 

# Kuromi images (place these files next to reactor.py)
IMG_ANGRY = "kuromi_angry.png"
IMG_AXE = "kuromi_axe.png"
IMG_DEFAULT = "kuromi_default.png"
IMG_HAPPY = "kuromi_happy.png"
IMG_TEASE = "kuromi_tease.png"

def load_and_resize(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    # Convert to BGR if image has alpha channel
    if img.shape[-1] == 4:
        # drop alpha or composite onto black background
        alpha = img[:, :, 3] / 255.0
        img = (img[:, :, :3].astype(np.float32) * alpha[:, :, None]).astype(np.uint8)
    img = cv2.resize(img, EMOJI_WINDOW_SIZE)
    return img

# Load images and handle missing files
try:
    kuromi_angry = load_and_resize(IMG_ANGRY)
    kuromi_axe = load_and_resize(IMG_AXE)
    kuromi_default = load_and_resize(IMG_DEFAULT)
    kuromi_happy = load_and_resize(IMG_HAPPY)
    kuromi_tease = load_and_resize(IMG_TEASE)
except Exception as e:
    print("Error loading Kuromi images:")
    print(e)
    print("Make sure the following files are present in the project directory:")
    print(", ".join([IMG_DEFAULT, IMG_HAPPY, IMG_ANGRY, IMG_AXE, IMG_TEASE]))
    raise SystemExit(1)

blank_emoji = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

def landmark_to_point(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def detect_axe_pose(pose_landmarks):
    # True if wrists are raised above shoulders OR both wrists are on the left side and
    # close together (side-by-side). Uses a small offset to avoid tiny variations.
    try:
        lw = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        rw = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        ls = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    except Exception:
        return False

    # If visibility attributes exist, ensure wrists are reasonably visible
    lw_vis = getattr(lw, 'visibility', 1.0)
    rw_vis = getattr(rw, 'visibility', 1.0)
    if lw_vis < 0.2 and rw_vis < 0.2:
        return False

    # Condition 1: one or both wrists raised above their respective shoulders
    # In MediaPipe coordinate system smaller y means higher in the image.
    left_raised = lw.y < (ls.y - RAISED_HAND_OFFSET)
    right_raised = rw.y < (rs.y - RAISED_HAND_OFFSET)
    if left_raised or right_raised:
        return True

    # Condition 2: two wrists are on the left side and close together (side-by-side)
    if lw.x < AXE_SIDE_X and rw.x < AXE_SIDE_X and abs(lw.x - rw.x) < AXE_MAX_X_DIFF:
        return True

    return False

def detect_smile(face_landmarks):
    # use same mouth corners and lips indices as the base example
    left_corner = face_landmarks.landmark[291]
    right_corner = face_landmarks.landmark[61]
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]

    mouth_width = ((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)**0.5
    mouth_height = ((lower_lip.x - upper_lip.x)**2 + (lower_lip.y - upper_lip.y)**2)**0.5
    if mouth_width <= 0:
        return False, 0.0 # Return MAR value for debugging
    mar = mouth_height / mouth_width
    
    is_smile = (SMILE_THRESHOLD < mar <= OPEN_MOUTH_THRESHOLD)
    
    return is_smile, mar # Return MAR value for debugging

def mouth_aspect_ratio(face_landmarks):
    left_corner = face_landmarks.landmark[291]
    right_corner = face_landmarks.landmark[61]
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    mouth_width = ((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)**0.5
    mouth_height = ((lower_lip.x - upper_lip.x)**2 + (lower_lip.y - upper_lip.y)**2)**0.5
    if mouth_width <= 0:
        return 0.0
    return mouth_height / mouth_width

def detect_tongue(frame, face_landmarks):
    h, w = frame.shape[:2]
    left_corner = face_landmarks.landmark[291]
    right_corner = face_landmarks.landmark[61]
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]

    x1 = int(min(left_corner.x, right_corner.x) * w)
    x2 = int(max(left_corner.x, right_corner.x) * w)
    y1 = int(max(0, (upper_lip.y - 0.02) * h))
    y2 = int(min(h, (lower_lip.y + 0.02) * h))

    if x2 - x1 < 5 or y2 - y1 < 5:
        return False, 0.0, 0.0 # Return MAR and red_frac for debugging

    mouth_roi = frame[y1:y2, x1:x2]
    if mouth_roi.size == 0:
        return False, 0.0, 0.0 # Return MAR and red_frac for debugging

    # Heuristic: tongue is reddish/pink. Use HSV mask for red/pink and require a minimum fraction of red pixels.
    hsv = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2HSV)
    # red can wrap hue, so check two ranges
    lower1 = np.array([0, 50, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 50, 50])
    upper2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    red_frac = (mask > 0).sum() / (mask.size)

    # also check mean R channel as backup
    mean_bgr = mouth_roi.reshape(-1, 3).mean(axis=0)
    mean_b, mean_g, mean_r = mean_bgr

    # require mouth to be relatively open (avoid false positives when lips closed)
    mar = mouth_aspect_ratio(face_landmarks)
    
    # --- MODIFICATION: Check if mouth is open enough for tongue ---
    if mar < OPEN_MOUTH_THRESHOLD:
        return False, mar, red_frac # Return values for debugging

    if (red_frac >= TONGUE_RED_FRAC and mean_r > mean_g * TONGUE_R_RATIO and mean_r > TONGUE_MIN_R):
        return True, mar, red_frac # Return values for debugging
    return False, mar, red_frac # Return values for debugging

def detect_angry(face_landmarks):
    # Heuristic: eyebrows lowered (approx) -> check vertical distance between eyebrow and eye
    # Use approximate indices for brow/eye in MediaPipe FaceMesh
    # These indices are heuristics and can be tuned if needed.
    left_brow = face_landmarks.landmark[70]
    left_eye_top = face_landmarks.landmark[159]
    right_brow = face_landmarks.landmark[300]
    right_eye_top = face_landmarks.landmark[386]

    brow_eye_dist_l = left_eye_top.y - left_brow.y
    brow_eye_dist_r = right_eye_top.y - right_brow.y

    # if the distance is small (brow close to eye), interpret as frown/angry
    # Use the new ANGRY_THRESHOLD
    is_angry = (brow_eye_dist_l < ANGRY_THRESHOLD or brow_eye_dist_r < ANGRY_THRESHOLD)

    return is_angry, brow_eye_dist_l, brow_eye_dist_r # Return values for debugging


def main():
    # --- TUNING: Make thresholds global so they can be modified ---
    global SMILE_THRESHOLD, OPEN_MOUTH_THRESHOLD, TONGUE_RED_FRAC, ANGRY_THRESHOLD

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow('Kuromi Reactor', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.resizeWindow('Kuromi Reactor', WINDOW_WIDTH, WINDOW_HEIGHT)

    print("--- LIVE TUNING CONTROLS ---")
    print("Press 'q' to quit")
    print("Smile Threshold (MAR):     'u' (up) / 'j' (down)")
    print("Tongue Threshold (MAR):    'i' (up) / 'k' (down)")
    print("Tongue Red Fraction:       'o' (up) / 'l' (down)")
    print("Angry Threshold (Brow):    'p' (up) / ';' (down)")
    print("----------------------------")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True) as face_mesh:

        # --- Initialize debug/tuning values ---
        debug_mar = 0.0
        debug_red_frac = 0.0
        debug_brow_l = 0.0
        debug_brow_r = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            debug_frame = frame.copy() 
            
            h, w = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            detected_state = 'DEFAULT'

            results_pose = pose.process(image_rgb)
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    debug_frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                if detect_axe_pose(results_pose.pose_landmarks):
                    detected_state = 'AXE'

            if detected_state != 'AXE':
                results_face = face_mesh.process(image_rgb)
                if results_face.multi_face_landmarks:
                    face_lms = results_face.multi_face_landmarks[0]

                    mp_drawing.draw_landmarks(
                        image=debug_frame, landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=debug_frame, landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=debug_frame, landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                    is_angry, debug_brow_l, debug_brow_r = detect_angry(face_lms)
                    is_smile, mar_smile = detect_smile(face_lms)
                    # --- Capture red_frac for HUD ---
                    is_tongue, mar_tongue, debug_red_frac = detect_tongue(frame, face_lms)
                    
                    debug_mar = max(mar_smile, mar_tongue) 

                    if is_angry:
                        detected_state = 'ANGRY'
                    elif is_tongue: # Give tongue priority over smile
                        detected_state = 'TEASE'
                    elif is_smile:
                        detected_state = 'HAPPY'
                    else:
                        detected_state = 'DEFAULT'
                
                else:
                    debug_mar, debug_red_frac, debug_brow_l, debug_brow_r = 0.0, 0.0, 0.0, 0.0
            
            # ----- Stability Smoothing -----
            try:
                counters
            except NameError:
                counters = {s: 0 for s in ('AXE', 'TEASE', 'HAPPY', 'ANGRY', 'DEFAULT')}
                display_state = 'DEFAULT'

            for s in counters:
                counters[s] = counters[s] + 1 if s == detected_state else 0

            STABILITY_FRAMES = 3
            for s, cnt in counters.items():
                if cnt >= STABILITY_FRAMES and display_state != s:
                    display_state = s
                    break

            # ----- Image Selection -----
            if display_state == 'AXE':
                out_img, label = kuromi_axe, 'AXE'
            elif display_state == 'TEASE':
                out_img, label = kuromi_tease, 'TEASE'
            elif display_state == 'HAPPY':
                out_img, label = kuromi_happy, 'HAPPY'
            elif display_state == 'ANGRY':
                out_img, label = kuromi_angry, 'ANGRY'
            else:
                out_img, label = kuromi_default, 'DEFAULT'

            # ----- Drawing and HUD -----
            camera_frame_resized = cv2.resize(debug_frame, (WINDOW_WIDTH, WINDOW_HEIGHT)) 

            state_color_map = {
                'AXE': (0, 165, 255), 'TEASE': (203, 192, 255), 'HAPPY': (0, 255, 0),
                'ANGRY': (0, 0, 255), 'DEFAULT': (255, 255, 255),
            }
            color = state_color_map.get(display_state, (255, 255, 255))

            # --- Main State Display ---
            cv2.putText(camera_frame_resized, f'STATE: {label}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            
            # --- Live Tuning HUD ---
            hud_y = 70
            # Smile
            cv2.putText(camera_frame_resized, f"[u/j] Smile MAR ({SMILE_THRESHOLD:.2f}): {debug_mar:.3f}", (10, hud_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if is_smile else (255,255,255), 2)
            # Tongue
            cv2.putText(camera_frame_resized, f"[i/k] Tongue MAR > {OPEN_MOUTH_THRESHOLD:.2f}", (10, hud_y + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(camera_frame_resized, f"[o/l] Tongue Red% ({TONGUE_RED_FRAC:.2f}): {debug_red_frac:.3f}", (10, hud_y + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if is_tongue else (255,255,255), 2)
            # Angry
            cv2.putText(camera_frame_resized, f"[p/;] Angry Brow < {ANGRY_THRESHOLD:.3f}: L={debug_brow_l:.3f} R={debug_brow_r:.3f}", (10, hud_y + 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if is_angry else (255,255,255), 2)

            cv2.putText(camera_frame_resized, 'Press "q" to quit', (10, WINDOW_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow('Camera Feed', camera_frame_resized)
            cv2.imshow('Kuromi Reactor', out_img)

            # --- Hotkey Logic ---
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            # Smile threshold
            elif key == ord('u'): SMILE_THRESHOLD += 0.01
            elif key == ord('j'): SMILE_THRESHOLD -= 0.01
            # Open mouth threshold
            elif key == ord('i'): OPEN_MOUTH_THRESHOLD += 0.01
            elif key == ord('k'): OPEN_MOUTH_THRESHOLD -= 0.01
            # Tongue red fraction
            elif key == ord('o'): TONGUE_RED_FRAC += 0.005
            elif key == ord('l'): TONGUE_RED_FRAC -= 0.005
            # Angry threshold
            elif key == ord('p'): ANGRY_THRESHOLD += 0.001
            elif key == ord(';'): ANGRY_THRESHOLD -= 0.001

    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Kuromi Reactor Closed ---")

if __name__ == '__main__':
    main()