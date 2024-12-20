import time
import cv2
from face_recognition.detector import FaceDetectorModel


def capture_photo(base_path, name, num_photos, camera_source):
    video_capture = cv2.VideoCapture(camera_source)  # Set the source webcam
    face_detector = FaceDetectorModel(cascade_path="../resources/models/faces/haarcascade_frontalface_default.xml")

    print("Enter 'c' to capture the photo.")
    print("Enter 'q' to quit...")
    print("Waiting to capture photo......\n\n")

    i = 0
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting..")
            break

        neram = str(int(time.time()))
        i = i + 1

        faces = face_detector.predict(gray)
        for j, face in enumerate(faces):
            (x, y, w, h) = face["pos"]
            face_img = face["img"]
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            cv2.imwrite(f"{base_path}/{name}.{neram}_{i}-{j}.png", face_img)

        cv2.imshow('Video', frame)
        cv2.waitKey(30)

        if i >= num_photos:
            break

        print("Saved as " + str(name) + "." + neram + ".png")
        
        # Add a 0.2-second delay to slow down the capture rate
        time.sleep(0.1)

    print(f"Record {num_photos} images. Process stopped.")
    video_capture.release()
    cv2.destroyAllWindows()
