import cv2
import os
import glob


class ImageHandler:

    def __init__(self, directory) -> None:
        self.directory = directory

    def open_image(self):
        # image_paths = os.path.join(self.directory, "*.jpg")
        image_paths = glob.glob(os.path.join(self.directory, "*.png"))
        for image_path in image_paths:
            print(image_path)
            img = cv2.imread(image_path)
            gray = self.__image_processing(img)
            cv2.imshow("image", gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def __image_processing(self, img) -> None:
        t1 = 30
        t2 = 100
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_img, (9, 9), 0)
        detected_edges = cv2.Canny(blur, t1, t2, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        close = cv2.morphologyEx(detected_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        circles = cv2.HoughCircles(
            close,
            cv2.HOUGH_GRADIENT,
            1.2,
            20,
            param1=50,
            param2=20,
            minRadius=1,
            maxRadius=20,
        )
        # print(circles)
        if circles is not None:
            print(len(circles))
            circles = circles[0, :]
            for i in circles:
                # draw the outer circle
                cv2.circle(gray_img, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(gray_img, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)

            print(len(circles))

        contours, hierarchy = cv2.findContours(
            close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        x0, y0, w0, h0 = cv2.boundingRect(contours[0])
        cv2.rectangle(gray_img, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 5)

        return gray_img


image_handler = ImageHandler("/home/michal/Desktop/dicestats/dice-stats/train/")

image_handler.open_image()
