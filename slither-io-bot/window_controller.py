import math, mss, cv2, pytesseract, time
import numpy as np

from selenium import webdriver
from selenium.webdriver.common.by import By
from pynput.mouse import Controller, Button
from pylsd.lsd import lsd

class WindowController:
    def __init__(self, url='http://www.slither.io', screen_capture=True) -> None:
        self.window_x = 0
        self.window_y = 0
        self.window_width = 600
        self.window_height = 600
        self.game_screen_x = 0
        self.game_screen_y = 80
        self.game_screen_center_x = self.game_screen_x + self.window_width / 2
        self.game_screen_center_y = self.game_screen_y + self.window_height / 2
        self.snake_head_x = self.game_screen_center_x
        # your character's head locates at a point slightly above the center of the game screen
        self.snake_head_y = self.game_screen_center_y - 25
        self.url = url
        self.mouse_controller = Controller()
        self.driver = None
        self.shot_count = 0
        self.screen_capture = screen_capture

    def get_snake_head(self) -> np.array:
        return np.array(([self.snake_head_x], [self.snake_head_y]))

    def open_browser(self) -> None:
        self.driver = webdriver.Safari()
        self.driver.set_window_size(self.window_width, self.window_height)
        self.driver.set_window_position(self.window_x, self.window_y)
        self.driver.get(self.url)

    def click(self, x, y, num_clicks=1) -> None:
        self.mouse_controller.position = (x, y)
        self.mouse_controller.click(Button.left, num_clicks)

    def click_play_game(self) -> None:
        start_button_x = self.game_screen_center_x
        start_button_y = self.window_y + 490
        self.click(start_button_x, start_button_y)

    def click_radian(self, radian, radius=150):
        self.click(
            self.snake_head_x + radius * math.cos(radian),
            self.snake_head_y + radius * math.sin(radian),
            2
        )

    def screenshot(self, x, y, width, height, reduction_factor=1, gray_scale=True):    
        sct = mss.mss()
        screen = {'left': x, 'top': y, 'width': width, 'height': height}
        # Grab the data
        img = sct.grab(screen)
        result = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2GRAY if gray_scale else cv2.COLOR_BGRA2BGR)
        img = result[::reduction_factor, ::reduction_factor]
        return img

    def read_score(self):
        score_x = 80
        score_y = 580# 590
        score_width = 80
        score_height = 30
        # self.driver.find_element(By.ID, "lastscore").rect
        return self.read_text(score_x, score_y, score_width, score_height)

    def read_last_score(self):
        last_score_x = 360
        last_score_y = 330
        return self.read_text(last_score_x, last_score_y, 100, 50)

    def read_text(self, x, y, width, height):
        image = self.screenshot(x, y, width, height)
        winname  = 'food'
        cv2.namedWindow(winname)        # Create a named window
        cv2.moveWindow(winname, 640,430)  # Move it to (40,30)
        cv2.imshow(winname, image)
        return pytesseract.image_to_string(image)

    def is_change_skin_visible(self):
        return self.driver.find_element(By.ID, "cskh").is_displayed()

    def get_last_score(self):
        print('get_last_score')
        text = self.driver.find_element(By.ID, "lastscore").value_of_css_property('innerText')
        print(text)

    def line_detector(self, image, canny_image):
        lines = lsd(canny_image)
        if self.shot_count < 1:
            self.shot_count = self.shot_count + 1
            cv2.imwrite('result{}.png'.format(self.shot_count), image)
        return lines

    def save_capture(self, image):
        cv2.imwrite('{}.png'.format(time.time()), image)

    def pellet_detector(self, canny_image):
        contours, _  = cv2.findContours(canny_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        pellets = []
        for c in contours:
            size = cv2.contourArea(c)
            if 0.0 < size < 1000:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                pellets.append((cX, cY))
        return pellets

    def get_canny_image(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny_image = cv2.Canny(gray_image, 150, 255)
        return canny_image

    def show_screen_capture(self, image, lines, pellets):
        if not self.screen_capture:
            return
        red = (0, 0, 255)
        blue = (255, 0, 0)
        for i in range(lines.shape[0]):
            pt1 = (int(lines[i, 0]), int(lines[i, 1]))
            pt2 = (int(lines[i, 2]), int(lines[i, 3]))
            width = lines[i, 4]
            cv2.line(image, pt1, pt2, red, int(np.ceil(width / 2)))
        for pellet in pellets:
            cv2.circle(image, pellet, 10, blue, -1)
        window_name  = 'capture'
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 600, 0)
        cv2.imshow(window_name, image)
        cv2.waitKey(25)

    def get_local_map(self):
        image = self.screenshot(self.game_screen_x, self.game_screen_y, self.window_width, self.window_height - 80, gray_scale=False)
        canny_image = self.get_canny_image(image)
        lines = self.line_detector(image, canny_image)
        pellets = self.pellet_detector(canny_image)
        self.show_screen_capture(image, lines, pellets)
        return lines, pellets, image
    
    @staticmethod
    def destroy_all_windows():
        cv2.destroyAllWindows()