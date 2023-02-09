from SCAutolib import run, logger

import os
from time import sleep, time
import cv2
import numpy as np
import pandas as pd
from io import StringIO
import pytesseract
import keyboard
import uinput
import functools


class Screen():
    """Captures the screenshots"""

    def __init__(self, directory: str):
        self.directory = directory
        self.screenshot_num = 1

    def screenshot(self):
        """
        Runs ffmpeg to take a screenshot.

        The filename of screenshot is then returned.
        """

        filename = f'/tmp/SC-tests/{self.screenshot_num}.png'
        run(['ffmpeg', '-hide_banner', '-y', '-f', 'kmsgrab', '-i', '-', '-vf',
             'hwdownload,format=bgr0', '-frames', '1', '-update', '1',
             filename])

        self.screenshot_num += 1
        return filename


class Mouse():
    def __init__(self):
        run(['modprobe', 'uinput'], check=True)
        # initialize the uinput device
        self.device = uinput.Device([
            uinput.REL_X,
            uinput.REL_Y,
            uinput.BTN_LEFT,
            uinput.BTN_MIDDLE,
            uinput.BTN_RIGHT,
            uinput.REL_WHEEL,
        ])

        self.MAX_RES = 10000
        self.FLICK_TIME = 0.1
        self.STEP_TIME = 0.005

    def move(self, x: int, y: int):
        """Moves the mouse cursor to specified absolute coordinate."""

        logger.info(f'Moving mouse to ({x, y})')

        for uinput_axis, value in [(uinput.REL_X, x), (uinput.REL_Y, y)]:
            # Go all the way up/left with the mouse
            sleep(self.FLICK_TIME)
            self.device.emit(uinput_axis, -self.MAX_RES)
            sleep(self.FLICK_TIME)

            # Go to the exact coordinate
            for i in range(value):
                self.device.emit(uinput_axis, 1)
                sleep(self.STEP_TIME)

    def click(self, button: str = 'left'):
        """
        Clicks the any button of the mouse.

        Button value can be 'left', 'right' or 'middle'.
        """

        button_map = {
            'left': uinput.BTN_LEFT,
            'right': uinput.BTN_RIGHT,
            'middle': uinput.BTN_MIDDLE,
        }
        uinput_button = button_map[button]

        logger.info(f'Clicking the {button} mouse button')

        # press the button
        self.device.emit(uinput_button, 1)
        # wait a little
        sleep(0.1)
        # release the button
        self.device.emit(uinput_button, 0)


class KB():
    """Wrapper class for keyboard library."""

    def __init__(self):
        def kb_decorator(fn):
            def wrapper(*args, **kwargs):
                logger.info('Using keyboard ...')  # TODO more info
                fn(*args, **kwargs)
                sleep(1)
            return wrapper

        # Workarounds for keyboard library
        # keyboard.write types nothing if the delay is not set
        self.write = kb_decorator(functools.partial(keyboard.write, delay=0.1))
        self.send = kb_decorator(keyboard.send)
        # For some reason the first keypress is never sent
        # So this effectively does nothing
        keyboard.send('enter')


class GUI():
    def __init__(self):
        # Create the directory for screenshots
        # TODO parametrize?
        self.screenshot_directory = '/tmp/SC-tests'
        os.makedirs(self.screenshot_directory, exist_ok=True)

        self.mouse = Mouse()

        self.kb = KB()

    def __enter__(self):
        self.screen = Screen(self.screenshot_directory)
        # By restarting gdm, the system gets into defined state
        run(['systemctl', 'restart', 'gdm'], check=True)
        # Cannot screenshot before gdm starts displaying
        # This would break the display
        sleep(5)

        return self

    def __exit__(self, type, value, traceback):
        # Gather the screenshots, generate report
        pass

    @staticmethod
    def _image_to_data(filename):
        """
        Convert screenshot into dataframe of words with their coordinates.
        """
        UPSCALING_FACTOR = 2

        image = cv2.imread(filename)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(grayscale,
                              dsize=None, fx=UPSCALING_FACTOR, fy=UPSCALING_FACTOR,
                              interpolation=cv2.INTER_LANCZOS4)
        _, binary = cv2.threshold(upscaled, 120, 255, cv2.THRESH_BINARY_INV)
        image_data_str = pytesseract.image_to_data(binary)
        df = pd.read_csv(StringIO(image_data_str),
                         sep='\t', lineterminator='\n')
        df[['left', 'top', 'width', 'height']] //= UPSCALING_FACTOR
        return df

    @staticmethod
    def _images_same(filename1: str, filename2: str):
        """Compare two images, return True if they are completely identical."""
        im1 = cv2.imread(filename1)
        im2 = cv2.imread(filename2)

        # Is the resolution the same
        if im1.shape != im2.shape:
            return False

        # Check if value of every pixel is the same
        if np.bitwise_xor(im1, im2).any():
            return False

        return True

    def click_on(self, key: str, timeout: float = 30):
        end_time = time() + timeout

        # Repeat screenshotting, until the key is found
        while time() < end_time:
            # Capture the screenshot
            screenshot = self.screen.screenshot()
            df = self._image_to_data(screenshot)
            logger.debug(df)
            selection = df['text'] == key

            # If there is no matching word, try again
            if selection.sum() == 0:
                logger.info('Found no match, trying again')
                continue

            # Exactly one word matching, exit the loop
            elif selection.sum() == 1:
                logger.info('Found exactly one match')
                item = df[selection]
                break

            # More than one word matches, choose the first match
            # Probably deterministic, but it should not be relied upon
            else:
                logger.info('Found multiple matches')
                item = df.iloc[0]

        if time() >= end_time:
            raise Exception('Found no matching key in any screenshot.')

        x = int(item['left'] + item['width']/2)
        y = int(item['top'] + item['height']/2)

        self.mouse.move(x, y)
        self.mouse.click()
        sleep(1)

    def assert_text(self, key: str, timeout: float = 0):
        """
        Given key must be found in a screenshot before the timeout.

        If the key is not found, exception is raised.
        Zero timeout means that only one screenshot
        will be taken and evaluated.
        """

        end_time = time() + timeout
        first = True

        while first or time() < end_time:
            first = False
            # Capture the screenshot
            screenshot = self.screen.screenshot()
            df = self._image_to_data(screenshot)
            selection = df['text'] == key

            # The key was found
            if selection.sum() != 0:
                return

        raise Exception('The key was not found.')

    def assert_no_text(self, key: str, timeout: float = 0):
        """
        If the given key is found in any screenshot before the timeout,
        an exception is raised.

        Zero timeout means that only one screenshot
        will be taken and evaluated.
        """

        end_time = time() + timeout
        first = True

        while first or time() < end_time:
            first = False
            # Capture the screenshot
            screenshot = self.screen.screenshot()
            df = self._image_to_data(screenshot)
            selection = df['text'] == key

            # The key was found, but should not be
            if selection.sum() != 0:
                raise Exception('The key was found in the screenshot.')

    def wait_still_screen(self, time_still: float = 5, timeout: float = 30):
        """
        Wait until the screen content stops changing.

        When nothing on screen has changed for time_still seconds, continue.
        If the screen content is changing permanently,
        fail with exception after timeout seconds.
        """

        timeout_end = time() + timeout
        time_still_end = time() + time_still
        last_screenshot = None

        while time() < timeout_end and time() < time_still_end:
            # Capture the screenshot
            screenshot = None
            while screenshot is None and time() < timeout_end:
                # If capturing fails, try again
                screenshot = self.screen.screenshot()

            # If this is the first loop, there is no last screenshot
            if last_screenshot is None:
                last_screenshot = screenshot
                time_still_end = time() + time_still
            # If the image has changed, refresh the time
            elif not self._images_same(screenshot, last_screenshot):
                last_screenshot = screenshot
                time_still_end = time() + time_still

        # If the loop was ended by timeout, it is an error
        if time() >= timeout_end:
            raise Exception('Screen contents were changing until timeout.')
