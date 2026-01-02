import unittest
import numpy as np
from unittest.mock import Mock

# Since we can't easily import from reactor.py without making it a package,
# and to keep this test self-contained, we will copy the detection functions
# and their dependencies here. In a larger project, you'd structure this as a package.

# --- Copied from reactor.py ---

# Configuration (tweak thresholds below if needed)
SMILE_THRESHOLD = 0.32
OPEN_MOUTH_THRESHOLD = 0.50
TONGUE_R_RATIO = 1.2
TONGUE_MIN_R = 80
TONGUE_RED_FRAC = 0.06
AXE_MAX_X_DIFF = 0.12
AXE_SIDE_X = 0.5
RAISED_HAND_OFFSET = 0.03

# Mock MediaPipe solution objects
mp_pose = Mock()
mp_pose.PoseLandmark = Mock()
# Assign integer values to landmarks for indexing
for i, name in enumerate(['LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_SHOULDER', 'RIGHT_SHOULDER']):
    setattr(mp_pose.PoseLandmark, name, i)


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

def detect_smile(face_landmarks):
    mar = mouth_aspect_ratio(face_landmarks)
    return (SMILE_THRESHOLD < mar <= OPEN_MOUTH_THRESHOLD)

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
        return False

    mouth_roi = frame[y1:y2, x1:x2]
    if mouth_roi.size == 0:
        return False

    # Heuristic: tongue is reddish/pink.
    # This is a simplified version of the logic in reactor.py for testing
    mean_bgr = mouth_roi.reshape(-1, 3).mean(axis=0)
    _, _, mean_r = mean_bgr
    
    mar = mouth_aspect_ratio(face_landmarks)
    if mar < OPEN_MOUTH_THRESHOLD:
        return False

    if mean_r > TONGUE_MIN_R:
        return True
    return False

def detect_angry(face_landmarks):
    left_brow = face_landmarks.landmark[70]
    left_eye_top = face_landmarks.landmark[159]
    right_brow = face_landmarks.landmark[300]
    right_eye_top = face_landmarks.landmark[386]

    brow_eye_dist_l = left_eye_top.y - left_brow.y
    brow_eye_dist_r = right_eye_top.y - right_brow.y

    if brow_eye_dist_l < 0.01 or brow_eye_dist_r < 0.01:
        return True
    return False

def detect_axe_pose(pose_landmarks):
    try:
        lw = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        rw = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        ls = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    except IndexError:
        return False

    lw_vis = getattr(lw, 'visibility', 1.0)
    rw_vis = getattr(rw, 'visibility', 1.0)
    if lw_vis < 0.2 and rw_vis < 0.2:
        return False

    left_raised = lw.y < (ls.y - RAISED_HAND_OFFSET)
    right_raised = rw.y < (rs.y - RAISED_HAND_OFFSET)
    if left_raised or right_raised:
        return True

    if lw.x < AXE_SIDE_X and rw.x < AXE_SIDE_X and abs(lw.x - rw.x) < AXE_MAX_X_DIFF:
        return True

    return False

# --- End of copied code ---


class MockLandmark:
    def __init__(self, x=0, y=0, z=0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

def create_mock_landmarks(points):
    """Creates a mock landmark list from a dictionary of index: (x, y) tuples."""
    landmarks = [MockLandmark() for _ in range(468)] # 468 landmarks for face mesh
    for idx, (x, y) in points.items():
        landmarks[idx] = MockLandmark(x, y)
    return Mock(landmark=landmarks)

def create_mock_pose_landmarks(points):
    """Creates a mock pose landmark list."""
    landmarks = [MockLandmark() for _ in range(33)] # 33 landmarks for pose
    for idx, (x, y, vis) in points.items():
        landmarks[idx] = MockLandmark(x, y, visibility=vis)
    return Mock(landmark=landmarks)


class TestReactorLogic(unittest.TestCase):

    def test_detect_smile(self):
        # Simulate a smile (mouth aspect ratio > SMILE_THRESHOLD)
        smile_points = {
            291: (0.3, 0.7), 61: (0.7, 0.7), # mouth corners
            13: (0.5, 0.65), 14: (0.5, 0.8) # lips (large height)
        }
        smile_landmarks = create_mock_landmarks(smile_points)
        self.assertTrue(detect_smile(smile_landmarks))

        # Simulate a neutral face
        neutral_points = {
            291: (0.3, 0.7), 61: (0.7, 0.7), # mouth corners
            13: (0.5, 0.74), 14: (0.5, 0.76) # lips (small height)
        }
        neutral_landmarks = create_mock_landmarks(neutral_points)
        self.assertFalse(detect_smile(neutral_landmarks))

    def test_detect_angry(self):
        # Simulate angry face (eyebrows close to eyes)
        angry_points = {
            70: (0.3, 0.3), 159: (0.3, 0.305), # left brow/eye
            300: (0.7, 0.3), 386: (0.7, 0.305) # right brow/eye
        }
        angry_landmarks = create_mock_landmarks(angry_points)
        self.assertTrue(detect_angry(angry_landmarks))

        # Simulate neutral face
        neutral_points = {
            70: (0.3, 0.3), 159: (0.3, 0.35), # left brow/eye
            300: (0.7, 0.3), 386: (0.7, 0.35) # right brow/eye
        }
        neutral_landmarks = create_mock_landmarks(neutral_points)
        self.assertFalse(detect_angry(neutral_landmarks))

    def test_detect_axe_pose_raised_hands(self):
        # Left hand raised
        points = {
            mp_pose.PoseLandmark.LEFT_WRIST: (0.2, 0.1, 0.9),
            mp_pose.PoseLandmark.LEFT_SHOULDER: (0.2, 0.4, 0.9),
            mp_pose.PoseLandmark.RIGHT_WRIST: (0.8, 0.6, 0.9),
            mp_pose.PoseLandmark.RIGHT_SHOULDER: (0.8, 0.4, 0.9),
        }
        landmarks = create_mock_pose_landmarks(points)
        self.assertTrue(detect_axe_pose(landmarks))

    def test_detect_axe_pose_side_by_side(self):
        # Hands side-by-side on the left
        points = {
            mp_pose.PoseLandmark.LEFT_WRIST: (0.2, 0.5, 0.9),
            mp_pose.PoseLandmark.RIGHT_WRIST: (0.3, 0.5, 0.9),
            mp_pose.PoseLandmark.LEFT_SHOULDER: (0.2, 0.3, 0.9),
            mp_pose.PoseLandmark.RIGHT_SHOULDER: (0.8, 0.3, 0.9),
        }
        landmarks = create_mock_pose_landmarks(points)
        self.assertTrue(detect_axe_pose(landmarks))

    def test_detect_axe_pose_neutral(self):
        # Hands down
        points = {
            mp_pose.PoseLandmark.LEFT_WRIST: (0.2, 0.6, 0.9),
            mp_pose.PoseLandmark.RIGHT_WRIST: (0.8, 0.6, 0.9),
            mp_pose.PoseLandmark.LEFT_SHOULDER: (0.2, 0.4, 0.9),
            mp_pose.PoseLandmark.RIGHT_SHOULDER: (0.8, 0.4, 0.9),
        }
        landmarks = create_mock_pose_landmarks(points)
        self.assertFalse(detect_axe_pose(landmarks))

    def test_detect_tongue(self):
        # Simulate open mouth
        open_mouth_points = {
            291: (0.3, 0.7), 61: (0.7, 0.7),
            13: (0.5, 0.6), 14: (0.5, 0.9) # very large height
        }
        landmarks = create_mock_landmarks(open_mouth_points)

        # Create a mock frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Simulate a red tongue in the mouth area
        tongue_frame = frame.copy()
        tongue_frame[65:85, 35:65] = (0, 0, 200) # BGR red color
        self.assertTrue(detect_tongue(tongue_frame, landmarks))

        # Simulate no tongue (black frame)
        self.assertFalse(detect_tongue(frame, landmarks))


if __name__ == '__main__':
    unittest.main()
