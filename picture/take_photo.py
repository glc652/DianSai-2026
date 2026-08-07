import cv2
from datetime import datetime
from pathlib import Path


SAVE_DIR = Path(r"E:\2026diansai\picture")
CAMERA_INDEX = 1


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Try changing CAMERA_INDEX to 1.")

    print("Camera opened.")
    print("Press SPACE to take a photo, or ESC/q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from camera.")
            break

        cv2.imshow("Camera - press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break

        if key == 32:
            filename = datetime.now().strftime("photo_%Y%m%d_%H%M%S.jpg")
            path = SAVE_DIR / filename
            cv2.imwrite(str(path), frame)
            print(f"Saved: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
