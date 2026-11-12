import cv2
from ultralytics import YOLO

def main():
    # fast model
    model=YOLO('yolov8s.pt')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Webcam not found / permission issue.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, device="mps", conf=0.2, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("Output", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()