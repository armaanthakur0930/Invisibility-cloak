import cv2
import numpy as np
import time

def capture_background(cap):
    print("Recapturing background. Please step out of the frame.")
    time.sleep(2) 
    _, background = cap.read()
    background = cv2.flip(background, 1)
    print("Background captured. You can step back in now.")
    return background

cap = cv2.VideoCapture(0)

print("Please wait while the camera initializes...")
for i in range(60):
    _, frame = cap.read()

background = capture_background(cap)

print("Press 'q' to quit.")

rgb_color = np.uint8([[[85, 64, 40]]])  
hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_BGR2HSV)
hue = hsv_color[0][0][0]

hue_range = 10
lower_bound = np.array([max(0, hue - hue_range), 50, 20])
upper_bound = np.array([min(180, hue + hue_range), 255, 255])

start_time = time.time()
background_capture_interval = 30 

while True:
    current_time = time.time()
    if current_time - start_time > background_capture_interval:
        background = capture_background(cap)
        start_time = current_time

    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
    
    mask_inv = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bg = cv2.bitwise_and(background, background, mask=mask)
    result = cv2.add(fg, bg)
    cv2.imshow("Invisibility Cloak", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()