# Kuromi Face Reactor

A real-time facial expression and gesture recognition system that displays Kuromi character reactions based on your facial expressions and hand gestures.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)

## 📋 Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Live Tuning](#live-tuning)
- [How It Works](#how-it-works)
- [Testing](#testing)
- [Contributing](#contributing)

## About

Kuromi Face Reactor is an interactive application that uses AI-powered facial recognition to detect your expressions and gestures, then displays corresponding Kuromi character reactions in real-time. It combines MediaPipe for facial/pose detection with OpenCV for video processing to create an engaging, responsive experience.

Perfect for:
- Fun interactive applications
- Social media integrations
- Stream overlays
- Educational demonstrations of computer vision

## Features

✨ **Emotion Detection**
- **Smile Detection**: Recognizes when you smile
- **Angry Face**: Detects when you furrow your brows
- **Tongue Out**: Identifies when you stick your tongue out
- **Axe Pose**: Detects when your hands are raised or held in specific positions

🎮 **Real-Time Processing**
- Live webcam feed with minimal latency
- Smooth state transitions with built-in jitter reduction
- Full HD-ready display

🛠️ **Developer-Friendly**
- Fully tunable detection thresholds
- Live HUD for real-time metric adjustment
- Comprehensive unit tests included
- Clean, well-documented codebase

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- 500MB free disk space for dependencies

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/SezarTheGreat/Kuromi-Reaction-for-emotions.git
   cd "Kuromi Reactor"
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv kuromienv
   kuromienv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv kuromienv
   source kuromienv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import cv2, mediapipe, numpy; print('All dependencies installed!')"
   ```

## Usage

### Running the Application

```bash
python reactor.py
```

The application will:
1. Open a window showing your webcam feed
2. Display the current emotion state in the top-left corner
3. Show the corresponding Kuromi image on the right side
4. Display detection metrics on the left side when in tuning mode

### Supported Emotions

| Expression | Trigger | Kuromi Image |
|-----------|---------|------------|
| **Default** | No emotion detected | `kuromi_default.png` |
| **Happy** | Smile detected (mouth open with smile shape) | `kuromi_happy.png` |
| **Angry** | Eyebrows furrowed (small brow-eye distance) | `kuromi_angry.png` |
| **Tease** | Tongue detected (red/pink pixels in mouth area) | `kuromi_tease.png` |
| **Axe** | Hands raised OR positioned side-by-side on left | `kuromi_axe.png` |

### Exiting

Press **`q`** to quit the application.

## Configuration

### Adjusting Thresholds

Open `reactor.py` and modify these constants to tune detection sensitivity:

```python
# Smile detection thresholds
SMILE_THRESHOLD = 0.32        # Min mouth aspect ratio for smile
OPEN_MOUTH_THRESHOLD = 0.50   # Max MAR before tongue detection kicks in

# Tongue detection parameters
TONGUE_RED_FRAC = 0.06        # % of mouth pixels that must be red (0.0-1.0)
TONGUE_MIN_R = 80             # Min red channel value (0-255)

# Angry face detection
ANGRY_THRESHOLD = 0.025       # Max brow-eye distance for anger

# Axe pose detection
RAISED_HAND_OFFSET = 0.03     # How much above shoulder = raised
AXE_MAX_X_DIFF = 0.12         # Max distance between wrists for side-by-side
AXE_SIDE_X = 0.5              # X threshold for left-side detection
```

**Tips for tuning:**
- Lower threshold values = more sensitive detection
- Higher values = require stronger/more obvious gestures
- Test with different lighting conditions for best results

## Live Tuning

While the app is running, use the **Live Tuning HUD** to adjust thresholds in real-time:

| Key | Action |
|-----|--------|
| `u` / `j` | Increase/decrease **Smile** MAR threshold |
| `i` / `k` | Increase/decrease **Tongue** MAR threshold |
| `o` / `l` | Increase/decrease **Tongue Red %** threshold |
| `p` / `;` | Increase/decrease **Angry** brow distance threshold |
| `q` | Quit application |

**Example:** Press `u` multiple times to make smile detection more sensitive.

The HUD displays:
- Current mouth aspect ratio (MAR)
- Red fraction percentage in mouth area
- Brow-eye distances for both sides
- All active threshold values

## How It Works

### Detection Pipeline

```
Webcam Frame
    ↓
[MediaPipe Face Mesh] → Extract 468 facial landmarks
    ↓
[MediaPipe Pose] → Extract 33 body landmarks
    ↓
[Detection Functions]
├─ Smile: MAR calculation from mouth landmarks
├─ Tongue: HSV color analysis of mouth ROI
├─ Angry: Eyebrow-eye vertical distance
└─ Axe: Wrist position relative to shoulders
    ↓
[State Smoothing] → Require 3 consecutive frames for state change
    ↓
[Display Manager] → Show corresponding Kuromi image
```

### Algorithm Details

- **Mouth Aspect Ratio (MAR)**: Calculates the ratio of mouth height to mouth width
- **HSV Color Detection**: Converts mouth region to HSV and identifies red/pink pixels
- **Facial Landmarks**: Uses 468-point face mesh for precise feature localization
- **Pose Estimation**: 33-point body skeleton for hand/shoulder tracking
- **Stability Buffer**: Prevents flickering by requiring consistent detection over 3 frames

## Testing

### Running Unit Tests

```bash
python -m unittest test_reactor.py -v
```

### Test Coverage

The test suite includes 6 comprehensive tests:

- `test_detect_smile`: Validates smile detection with mock landmarks
- `test_detect_tongue`: Tests red pixel detection in mouth region
- `test_detect_angry`: Verifies eyebrow furrow detection
- `test_detect_axe_pose_raised_hands`: Tests hand-raise detection
- `test_detect_axe_pose_side_by_side`: Validates side-by-side hand detection
- Additional edge case validations

**Sample output:**
```
test_detect_angry (test_reactor.TestReactorLogic) ... ok
test_detect_axe_pose_raised_hands (test_reactor.TestReactorLogic) ... ok
test_detect_axe_pose_side_by_side (test_reactor.TestReactorLogic) ... ok
test_detect_smile (test_reactor.TestReactorLogic) ... ok
test_detect_tongue (test_reactor.TestReactorLogic) ... ok

Ran 6 tests in 0.045s
OK
```

## Project Structure

```
Kuromi Reactor/
├── reactor.py                 # Main application
├── test_reactor.py            # Unit tests
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── kuromi_*.png              # Kuromi character images
    ├── kuromi_default.png     # Neutral expression
    ├── kuromi_happy.png       # Smiling
    ├── kuromi_angry.png       # Angry/upset
    ├── kuromi_tease.png       # Tongue out
    └── kuromi_axe.png         # Axe pose
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.8.0 | Video capture & image processing |
| mediapipe | ≥0.10.13 | Facial & pose landmark detection |
| numpy | ≥1.24.0 | Numerical operations |
| Pillow | ≥10.0.0 | Image handling |

## System Requirements

- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **RAM**: 4GB minimum (8GB recommended)
- **CPU**: Modern multi-core processor
- **GPU**: Optional (improves performance ~2-3x with CUDA)

## Performance Tips

- Use a well-lit environment for best detection
- Position camera at face level, 60-80cm away
- Keep the webcam feed unobstructed
- Close unnecessary applications to free up system resources

## Troubleshooting

### Camera not detected
```bash
# Verify camera access (Windows)
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera not found')"
```

### Low detection accuracy
- Increase lighting in your environment
- Clean your webcam lens
- Adjust thresholds using live tuning mode (press `u`, `i`, `o`, `p` keys)
- Test with exaggerated expressions

### Performance issues
- Reduce window size in code (adjust `WINDOW_WIDTH`, `WINDOW_HEIGHT`)
- Use GPU acceleration if available
- Close other applications consuming system resources

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for facial and pose detection models
- [OpenCV](https://opencv.org/) for computer vision utilities
- Kuromi character assets and inspiration

## Support

For issues, questions, or suggestions:
- Open an [issue on GitHub](https://github.com/SezarTheGreat/Kuromi-Reaction-for-emotions/issues)
- Check existing issues for solutions

## Future Enhancements

Potential improvements for future versions:
- [ ] Multi-face detection support
- [ ] Recording and playback functionality
- [ ] Performance metrics and FPS counter
- [ ] Custom emotion configuration
- [ ] Mobile/web deployment options
- [ ] Alternative character sets

---

**Happy reacting! 🎉**

*Built with ❤️ using MediaPipe and OpenCV*
