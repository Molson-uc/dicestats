import cv2 as cv
import camera.CameraHandler as ch
import time

if __name__ == "__main__":
    cam = ch.CameraHandler(0)
    while True:
        cam.show()
        if cv.waitKey(1) == ord("q"):
            cam.close()
            break
