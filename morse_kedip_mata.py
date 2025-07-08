import cv2
import dlib
import time
import numpy as np
from collections import deque
from scipy.spatial import distance as dist

class EyeBlinkMorseDecoder:
    # Dictionary Morse code
    MORSE_CODE_DICT = { 'A':'.-', 'B':'-...',
                        'C':'-.-.', 'D':'-..', 'E':'.',
                        'F':'..-.', 'G':'--.', 'H':'....',
                        'I':'..', 'J':'.---', 'K':'-.-',
                        'L':'.-..', 'M':'--', 'N':'-.',
                        'O':'---', 'P':'.--.', 'Q':'--.-',
                        'R':'.-.', 'S':'...', 'T':'-',
                        'U':'..-', 'V':'...-', 'W':'.--',
                        'X':'-..-', 'Y':'-.--', 'Z':'--..',
                        '1':'.----', '2':'..---', '3':'...--',
                        '4':'....-', '5':'.....', '6':'-....',
                        '7':'--...', '8':'---..', '9':'----.',
                        '0':'-----', ', ':'--..--', '.':'.-.-.-',
                        '?':'..--..', '/':'-..-.', '-':'-....-',
                        '(':'-.--.', ')':'-.--.-', ' ':' '}

    def __init__(self):
        # Initialize dlib's face detector and facial landmarks predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Eye aspect ratio thresholds
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES_CLOSED = 2
        self.EYE_AR_CONSEC_FRAMES_OPEN = 3
        
        # Timing parameters (in seconds)
        self.dot_threshold = 0.3  # Default threshold for dot/dash classification
        self.letter_pause = 1.0    # Time between letters
        self.word_pause = 3.0      # Time between words
        
        # State tracking
        self.current_symbol = ''
        self.current_word = ''
        self.message = ''
        self.last_blink_end = 0
        self.blink_start = 0
        self.eye_closed_frames = 0
        self.eye_open_frames = 0
        self.blinking = False
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        # Visualization parameters
        self.COLOR_FACE = (255, 0, 0)     # Blue
        self.COLOR_EYES = (0, 255, 0)     # Green
        self.COLOR_TEXT = (0, 0, 255)     # Red
        
        # Blink history for adaptive threshold
        self.blink_history = deque(maxlen=10)
    
    def eye_aspect_ratio(self, eye):
        # Compute the Euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # Compute the Euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        return ear
    
    def decode_symbol(self, symbol):
        """Convert Morse code to character"""
        reversed_dict = {v: k for k, v in self.MORSE_CODE_DICT.items()}
        return reversed_dict.get(symbol, None)
    
    def process_frame(self, frame):
        """Process each frame for eye detection and blink analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        rects = self.detector(gray, 0)
        eye_status = "Open"
        
        for rect in rects:
            # Get facial landmarks
            shape = self.predictor(gray, rect)
            shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
            # Extract eye landmarks
            left_eye = shape[42:48]
            right_eye = shape[36:42]
            
            # Compute eye aspect ratios
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            
            # Average the eye aspect ratio together for both eyes
            ear = (left_ear + right_ear) / 2.0
            
            # Check for blinking
            if ear < self.EYE_AR_THRESH:
                self.eye_closed_frames += 1
                self.eye_open_frames = 0
                eye_status = "Closed"
            else:
                self.eye_open_frames += 1
                if self.eye_open_frames >= self.EYE_AR_CONSEC_FRAMES_OPEN:
                    self.eye_closed_frames = 0
                    eye_status = "Open"
            
            # Check for blink state changes
            self.check_blink(eye_status)
            
            # Draw eye contours
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, self.COLOR_EYES, 1)
            cv2.drawContours(frame, [right_eye_hull], -1, self.COLOR_EYES, 1)
            
        # Add status text to frame
        cv2.putText(frame, f"Eye: {eye_status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)
        
        # Add Morse code status to frame
        cv2.putText(frame, f"Symbol: {self.current_symbol}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)
        cv2.putText(frame, f"Word: {self.current_word}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)
        cv2.putText(frame, f"Message: {self.message}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)
        
        return frame
    
    def check_blink(self, current_status):
        """Analyze blink state and convert to Morse code"""
        current_time = time.time()
        
        # Detect blink start (transition from open to closed)
        if not self.blinking and current_status == "Closed" and self.eye_closed_frames >= self.EYE_AR_CONSEC_FRAMES_CLOSED:
            self.blinking = True
            self.blink_start = current_time
        
        # Detect blink end (transition from closed to open)
        elif self.blinking and current_status == "Open" and self.eye_open_frames >= self.EYE_AR_CONSEC_FRAMES_OPEN:
            self.blinking = False
            blink_duration = current_time - self.blink_start
            self.blink_history.append(blink_duration)
            
            # Update threshold dynamically based on recent blinks
            if len(self.blink_history) > 3:
                avg_blink = sum(self.blink_history) / len(self.blink_history)
                self.dot_threshold = avg_blink * 0.8  # 80% of average as threshold
            
            # Classify as dot or dash
            if blink_duration < self.dot_threshold:
                self.current_symbol += '.'
            else:
                self.current_symbol += '-'
            
            self.last_blink_end = current_time
        
        # Check for timeouts between symbols/words
        elif not self.blinking and self.last_blink_end > 0:
            time_since_last_blink = current_time - self.last_blink_end
            
            # Symbol timeout
            if time_since_last_blink > self.letter_pause and self.current_symbol:
                char = self.decode_symbol(self.current_symbol)
                if char:
                    self.current_word += char
                    self.message += char
                else:
                    self.message += '<?>'
                self.current_symbol = ''
                
                # Word timeout
                if time_since_last_blink > self.word_pause and self.current_word:
                    self.message += ' '
                    self.current_word = ''
    
    def run(self):
        """Main loop for video capture and processing"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)  # Mirror image for more natural interaction
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow('Eye Blink Morse Decoder', processed_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Silahkan berkedip")
    print("kedip cepat <0.3 detik = .")
    print("kedip lambat >0.3 detik = -")
    decoder = EyeBlinkMorseDecoder()
    decoder.run()
