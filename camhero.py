# Step 1: Import necessary libraries
import cv2
import numpy as np
import pygame
import time
import random


# Step 2: Set up the webcam and video capture
def setup_webcam():
    cap = cv2.VideoCapture(0)
    return cap

# Step 3: Define regions for movement detection
def define_regions(frame):
    height, width, _ = frame.shape
    size = min(width, height) // 4
    regions = [
        {'name': 'C', 'rect': (width//2 - size//2, 0, size, size)},         # Top
        {'name': 'D', 'rect': (width//2 - size//2, height-size, size, size)},  # Bottom
        {'name': 'E', 'rect': (0, height//2 - size//2, size, size)},       # Left
        {'name': 'F', 'rect': (width-size, height//2 - size//2, size, size)}  # Right
    ]
    return regions


# Step 4: Process each video frame to detect movement in the specified regions
def detect_movement(frame, prev_frame, regions, threshold=40):
    movement_detected = []
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    gray_prev_frame = cv2.GaussianBlur(gray_prev_frame, (21, 21), 0)

    for region in regions:
        x, y, w, h = region['rect']
        diff = cv2.absdiff(gray_frame[y:y+h, x:x+w], gray_prev_frame[y:y+h, x:x+w])
        non_zero_count = np.count_nonzero(diff > threshold)

        if non_zero_count > 2000:  # Increase this value to require more significant movement for detection
            movement_detected.append(region['name'])

    return movement_detected


# Step 5: Map the detected movement to musical notes
def map_movement_to_notes(movement_detected):
    note_mapping = {
        'C': 'C4',
        'D': 'D4',
        'E': 'E4',
        'F': 'F4'
    }

    notes_to_play = [note_mapping[note] for note in movement_detected]
    return notes_to_play

import random




# Step 6: Play the corresponding musical notes
def play_notes(notes_to_play):
    pygame.mixer.init()

    for note in notes_to_play:
        sound_file = f"notes/{note}.mp3"
        sound = pygame.mixer.Sound(sound_file)
        sound.play()


# Initialize the list of symbols
def generate_random_symbols(num_symbols=10, last_start=0):
    symbols = []
    for i in range(num_symbols):
        region = random.choice(['C', 'D', 'E', 'F'])
        start = random.uniform(last_start + i * 5, last_start + i * 5 + 5)
        end = start + 1
        symbols.append({'region': region, 'start': start, 'end': end})
    return symbols

SYMBOLS = generate_random_symbols()

# Initialize the score
score = 0

missed_symbols = 0

def draw_symbols(frame, regions, elapsed_time):
    global SYMBOLS

    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    for symbol in SYMBOLS:
        if symbol['start'] <= elapsed_time <= symbol['end']:
            for region in regions:
                if region['name'] == symbol['region']:
                    x, y, w, h = region['rect']
                    progress = (elapsed_time - symbol['start']) / (symbol['end'] - symbol['start'])
                    symbol_x = int(center_x + progress * (x + w//2 - center_x))
                    symbol_y = int(center_y + progress * (y + h//2 - center_y))
                    radius = min(w, h) // 3
                    cv2.circle(frame, (symbol_x, symbol_y), radius, (0, 0, 255), 2)
                    break

def update_score(movement_detected, elapsed_time):
    global SYMBOLS, score, missed_symbols
    for symbol in SYMBOLS:
        if symbol['start'] <= elapsed_time <= symbol['end']:
            if symbol['region'] in movement_detected:
                score += 1
                # Remove the symbol from the list
                SYMBOLS = [s for s in SYMBOLS if s != symbol]
                # Generate a new symbol and append it to the list
                new_symbols = generate_random_symbols(1, elapsed_time)
                SYMBOLS.extend(new_symbols)
        elif elapsed_time > symbol['end']:
            missed_symbols += 1
            # Remove the symbol from the list
            SYMBOLS = [s for s in SYMBOLS if s != symbol]
            # Generate a new symbol and append it to the list
            new_symbols = generate_random_symbols(1, elapsed_time)
            SYMBOLS.extend(new_symbols)


# Main function to execute the steps
def main():
    global score, missed_symbols
    cap = setup_webcam()
    ret, prev_frame = cap.read()

    # Record the start time of the game
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip the camera horizontally
        elapsed_time = time.time() - start_time

        regions = define_regions(frame)
        movement_detected = detect_movement(frame, prev_frame, regions)
        notes_to_play = map_movement_to_notes(movement_detected)
        play_notes(notes_to_play)

        draw_symbols(frame, regions, elapsed_time)
        update_score(movement_detected, elapsed_time)

        # Display the regions, score, and video frame
        for region in regions:
            x, y, w, h = region['rect']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, region['name'], (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        #display the score and missed symbols on the frame
        cv2.putText(frame, "Score: " + str(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Missed symbols: " + str(missed_symbols), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)        

        cv2.imshow('Guitar Hero Movement Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or missed_symbols == 5:
            break

        prev_frame = frame.copy()

    print("Game Over! Your score:", score)
    print("Missed symbols:", missed_symbols)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
