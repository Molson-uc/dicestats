import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import imutils


class CameraHandler:

    def __init__(self, camera_id) -> None:
        self.capture = cv.VideoCapture(camera_id)

        def nothing(x):
            pass

        cv.namedWindow("image")
        cv.createTrackbar("t1", "image", 0, 255, nothing)
        cv.createTrackbar("t2", "image", 0, 255, nothing)

        cv.createTrackbar("b1", "image", 1, 255, nothing)
        cv.createTrackbar("ker", "image", 1, 20, nothing)
        cv.createTrackbar("dp", "image", 0, 255, nothing)
        cv.createTrackbar("minD", "image", 0, 200, nothing)
        cv.createTrackbar("p1", "image", 0, 200, nothing)
        cv.createTrackbar("p2", "image", 0, 200, nothing)

    def __is_opened(self) -> bool:
        if not self.capture.isOpened():
            print("Cannot open camera")
            exit()
        return True

    def _capture_frame(self) -> cv.typing.MatLike | None:
        self.__is_opened()
        ret, frame = self.capture.read()
        if ret:
            # mod_frame = self.__modify_frame(frame)
            f = self.__recognization(frame)
            return f
        return None

    def __modify_frame(self, frame) -> cv.typing.MatLike:
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return gray_frame

    def show(self) -> None:
        frame = self._capture_frame()
        cv.imshow("frame", frame)

    def close(self) -> None:
        self.capture.release()
        cv.destroyAllWindows()

    def __recognization(self, gray_img) -> cv.typing.MatLike:

        t1 = cv.getTrackbarPos("t1", "image")
        t2 = cv.getTrackbarPos("t2", "image")
        b1 = cv.getTrackbarPos("b1", "image")
        ker = cv.getTrackbarPos("ker", "image")
        dp = cv.getTrackbarPos("dp", "image")
        minD = cv.getTrackbarPos("minD", "image")
        p1 = cv.getTrackbarPos("p1", "image")
        p2 = cv.getTrackbarPos("p2", "image")

        blur = cv.GaussianBlur(gray_img, (5, 5), 0)
        detected_edges = cv.Canny(blur, t1, t2, 3)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))

        close = cv.morphologyEx(detected_edges, cv.MORPH_CLOSE, kernel, iterations=1)
        circles = cv.HoughCircles(
            close,
            cv.HOUGH_GRADIENT,
            1.2,
            30,
            param1=100,
            param2=50,
            minRadius=1,
            maxRadius=50,
        )
        # print(circles)
        if circles is not None:
            circles = circles[0, :]
            for i in circles:
                # draw the outer circle
                cv.circle(gray_img, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
                # draw the center of the circle
                cv.circle(gray_img, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)

            print(len(circles))

        contours, hierarchy = cv.findContours(
            close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        x0, y0, w0, h0 = cv.boundingRect(contours[0])
        cv.rectangle(gray_img, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 5)

        return close
