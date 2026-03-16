from ultralytics import YOLO
import cv2

# YOLO Modell laden
model = YOLO("yolov8n.pt")

# Webcam starten
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Objekterkennung
    results = model(frame)

    # Ergebnisse anzeigen
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Programm beenden mit Taste q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
