import cv2
import threading
import time


class LatestFrameReader:
    def __init__(self, cam_url):
        self.cap = cv2.VideoCapture(cam_url)
        if not self.cap.isOpened():
            raise RuntimeError('Cannot open RTSP stream of the camera')

        self.lock = threading.Lock()
        self.frame = None
        self.running = True

        self.thread = threading.Thread(
            target=self._reader,
            daemon=True
        )
        self.thread.start()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()