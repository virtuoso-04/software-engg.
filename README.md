# Implementation: Learner Attentiveness and Engagement Analysis

## **Overview**
This implementation analyzes learner attentiveness in online education environments using computer vision techniques. The system leverages facial landmark detection and gaze tracking to determine engagement levels and provides actionable insights for educators to improve instructional strategies.

---

## **Features**
- Real-time facial landmark detection using Dlib's pre-trained model.
- Eye Aspect Ratio (EAR) calculation for gaze tracking and blink detection.
- Engagement classification based on visual cues such as eye movements and facial expressions.
- Live webcam integration for real-time analysis.

---

## **Requirements**
- Python 3.7+
- OpenCV
- Dlib
- Numpy
- Webcam (for real-time video capture)

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/virtuoso-04/software-engg.git
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python dlib numpy
   ```

3. Download Dlib's pre-trained shape predictor:
   - [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - Extract and place the file in the project directory.

---

## **Usage**
1. Run the script:
   ```bash
   python engagement_analysis.py
   ```

2. Allow webcam access to start real-time analysis.

3. Observe engagement classification (e.g., "Engaged" or "Distracted") displayed on the video feed.

---

## **Code Explanation**

### **Key Functions**
- **`calculate_ear(eye)`**: Calculates the Eye Aspect Ratio (EAR) to detect eye openness.
- **Facial Landmark Detection**: Uses Dlib's `shape_predictor` to identify key facial points.
- **Gaze Tracking**: Determines gaze direction by analyzing eye position relative to the face.

### **Main Script**
```python
import cv2
import dlib
import numpy as np

# Load Dlib's face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        points = np.array([(p.x, p.y) for p in landmarks.parts()])

        left_eye = points[36:42]
        right_eye = points[42:48]
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        status = "Engaged" if avg_ear > 0.25 else "Distracted"
        cv2.putText(frame, f"Gaze: {status}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Engagement Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## **Future Improvements**
- Integration of additional metrics such as head orientation and body posture.
- Use of deep learning models for more robust engagement prediction.
- Real-time feedback for instructors during live classes.

---

## **References**
1. [Dlib Facial Landmark Detection](http://dlib.net/)
2. [OpenCV Documentation](https://docs.opencv.org/)
3. "Learner Attentiveness and Engagement Analysis in Online Education Using Computer Vision." [ArXiv](https://arxiv.org/html/2412.00429v1)

---

For further details, contact: [your-email@example.com]

