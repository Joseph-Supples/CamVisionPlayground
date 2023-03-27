import numpy as np
import time
import cv2
import pyaudio 

def setup_webcam():
    cap = cv2.VideoCapture(0)
    return cap
def detect_hand_position(frame, prev_frame):
    diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is not None and max_area > 1000:
        x, y, w, h = cv2.boundingRect(max_contour)
        return x, y, w, h

    print("Hand not detected:", frame.shape, prev_frame.shape, diff.shape, thresh.shape)  # Add this line
    return 0, 0



def position_to_pitch_and_vibrato(pos, frame_width, frame_height):
    if pos is not None:
        cX, cY = pos
        pitch = int(np.interp(cY, [0, frame_height], [2000, 200]))  # Map Y position to pitch (200Hz - 2000Hz)
        vibrato = int(np.interp(cX, [0, frame_width], [0, 1000]))  # Map X position to vibrato (0 - 100)
        return pitch, vibrato
    else:
        return None, None
def setup_tracker(cap):
    attempts = 0
    max_attempts = 10
    tracker = None

    while attempts < max_attempts:
        _, prev_frame = cap.read()
        time.sleep(0.05)  # Small delay to capture a slightly different frame
        _, frame = cap.read()
        
        hand_position = detect_hand_position(frame, prev_frame)
        print("Hand position:", hand_position)  # Add this line

        if hand_position and len(hand_position) == 4:
            x, y, w, h = hand_position
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, tuple(map(int, (x, y, w, h))))
            break

        attempts += 1

    return tracker, frame





current_pitch_vibrato = (0, 0)
vibrato_intensity = 2
volume = 0.8

def play_theremin_synth():
    global current_pitch_vibrato
    sample_rate = 44100
    buffer_size = 1024
    p = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        global current_pitch_vibrato, vibrato_intensity, volume
        pitch, vibrato = current_pitch_vibrato
        if pitch is None:
            return (in_data, pyaudio.paComplete)

        duration = frame_count / sample_rate
        t = np.linspace(0, duration, frame_count, False)
        sine_wave = (
            np.sin((pitch - vibrato * vibrato_intensity) * t * 2 * np.pi)
            + np.sin((pitch + vibrato * vibrato_intensity) * t * 2 * np.pi)
        ) / 2

        audio_data = (sine_wave * volume * 32767 / np.max(np.abs(sine_wave))).astype(np.int16)
        return (audio_data.tobytes(), pyaudio.paContinue)

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True, stream_callback=callback)

    stream.start_stream()
    return stream, p


def main():
    global current_pitch_vibrato
    cap = setup_webcam()
    tracker, prev_frame = setup_tracker(cap)

    if tracker is None:
        print("Failed to initialize tracker. Please try again.")
        return

    stream, p = play_theremin_synth()

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip the camera horizontally
        height, width, _ = frame.shape
        ret, frame = cap.read()

        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = tuple(map(int, bbox))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            current_pitch_vibrato = position_to_pitch_and_vibrato((x + w // 2, y + h // 2), width, height)


        # hand_position = detect_hand_position(frame, prev_frame)
        # if hand_position:
        #     cv2.circle(frame, hand_position, 10, (0, 255, 0), -1)  # Draw a circle at the hand's position
        #     current_pitch_vibrato = position_to_pitch_and_vibrato(hand_position, width, height)


        cv2.imshow('Theremin Movement Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = frame.copy()

    current_pitch_vibrato = (None, None)  # Signal the audio thread to exit
    stream.stop_stream()
    stream.close()
    p.terminate()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()