import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables
zoom_level = 1
zoom_enabled = False
debug_enabled = False
face_size = 1
cx = 0
cy = 0

while True:
    
    # Handle keyboard
    pressedKey = cv2.waitKey(1)
    if pressedKey == ord('q'):
        break
    elif pressedKey == ord('z'):
        zoom_enabled = not zoom_enabled  
    elif pressedKey == ord('d'):
        debug_enabled = not debug_enabled
    
    # Obtain video information
    ret, frame = cap.read()
    height, width, channels = frame.shape 
    tickmark=cv2.getTickCount()
    
    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 10)
    
    for (x, y, w, h) in faces:
        # Size of the face
        face_size = h
        
        # Center of the face
        cx = int(x + w/2)
        cy = int(y + h/2)
        
        # Draw a rectangle around the face and its center
        if debug_enabled:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(frame, (cx, cy), (cx, cy), (0, 0, 255), 2)
    
    # Compute the maximum zoom
    max_zoom = 0.8*height/face_size
    
    # Hysteresis factor in the zoom level, for more stability
    hysteresis = 0.3
    # Zoom in or zoom out depending on the situation
    if len(faces)>0 and zoom_level<max_zoom and zoom_enabled:
        zoom_level += 0.1
    elif (len(faces)==0 and zoom_level>1) or (len(faces)>0 and zoom_level>max_zoom+hysteresis and zoom_level>1) or (not zoom_enabled and zoom_level>1):
        zoom_level -= 0.1 
    # Avoid issues due  to float precision
    if zoom_level<1:
        zoom_level = 1
    
    # Compute coordinates of the zoomed region
    x1, x2, y1, y2 = 0, width, 0, height
    x1 = int( cx + (x1-cx)/zoom_level )
    x2 = int( cx + (x2-cx)/zoom_level )
    y1 = int( cy + (y1-cy)/zoom_level )
    y2 = int( cy + (y2-cy)/zoom_level )
    
    # Crop and resize the frame
    cropped_frame = frame[y1:y2, x1:x2]
    resized_frame = cv2.resize(cropped_frame, (width, height)) 
    
    # Print information on the frame
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(resized_frame, "{:05.2f}".format(fps)+"fps", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(resized_frame, "zoom:"+"{:.1f}".format(zoom_level), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if zoom_enabled:
        cv2.putText(resized_frame, "zoom ON",   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(resized_frame, "zoom OFF",  (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if debug_enabled:
        cv2.putText(resized_frame, "debug ON",  (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(resized_frame, "debug OFF", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Automatic zoom', resized_frame)

cap.release()
cv2.destroyAllWindows()