import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Testing camera FPS for 10 seconds...")
print("-" * 40)

frame_count = 0
start_time = time.time()

while time.time() - start_time < 10:
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        # Display frame with FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('FPS Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
cap.release()
cv2.destroyAllWindows()

elapsed = end_time - start_time
avg_fps = frame_count / elapsed

print(f"Frames captured: {frame_count}")
print(f"Time elapsed: {elapsed:.2f} seconds")
print(f"Average FPS: {avg_fps:.2f}")