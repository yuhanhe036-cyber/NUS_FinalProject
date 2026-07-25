import cv2

# Open webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow('Face Keypoints', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Keypoints', 640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Face Keypoints', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
