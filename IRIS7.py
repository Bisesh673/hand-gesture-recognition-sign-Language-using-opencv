import cv2
import numpy as np
from collections import deque

finger_history = deque(maxlen=10)  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

   
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin_ycrcb = np.array([0, 133, 77], np.uint8)
    upper_skin_ycrcb = np.array([255, 173, 127], np.uint8)

    lower_skin_hsv = np.array([0, 30, 60], np.uint8)
    upper_skin_hsv = np.array([20, 150, 255], np.uint8)

    mask_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
    mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)

    mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 3000: 
            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)

            hull = cv2.convexHull(cnt)
            cv2.drawContours(roi, [hull], -1, (0, 0, 255), 2)

            hull_indices = cv2.convexHull(cnt, returnPoints=False)
            if len(hull_indices) > 3:
                defects = cv2.convexityDefects(cnt, hull_indices)
                if defects is not None:
                    count_defects = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start, end, far = tuple(cnt[s][0]), tuple(cnt[e][0]), tuple(cnt[f][0])

                        
                        a = np.linalg.norm(np.array(end) - np.array(start))
                        b = np.linalg.norm(np.array(far) - np.array(start))
                        c = np.linalg.norm(np.array(end) - np.array(far))
                        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c + 1e-5)) * 57

                        
                        if angle <= 85 and d > 12000 and a > 40:
                            count_defects += 1
                            cv2.circle(roi, far, 5, (255, 0, 0), -1)

                    # Map finger counts
                    if count_defects == 0:
                        gesture = "Fist / 1"
                    elif count_defects == 1:
                        gesture = "2 Fingers"
                    elif count_defects == 2:
                        gesture = "3 Fingers"
                    elif count_defects == 3:
                        gesture = "4 Fingers"
                    elif count_defects >= 4:
                        gesture = "Open Palm"
                    else:
                        gesture = "Unknown"

                    
                    finger_history.append(gesture)
                    stable_gesture = max(set(finger_history), key=finger_history.count)

                    cv2.putText(frame, f"Gesture: {stable_gesture}", (50, 450),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()