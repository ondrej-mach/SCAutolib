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
    """Captures the screenshots."""

    def __init__(self, directory: str):
        self.directory = directory
        self.screenshot_num = 1

    def screenshot(self, timeout: float = 30):
        """Runs ffmpeg to take a screenshot.

        :return: filename of the screenshot
        :rtype: str
        """

        logger.debug(f"Taking screenshot number {self.screenshot_num}")

        filename = f'{self.directory}/{self.screenshot_num}.png'
        t_end = time() + timeout
        captured = False

        while time() < t_end and not captured:
            out = run(['ffmpeg', '-hide_banner', '-y', '-f',
                       'kmsgrab', '-i', '-', '-vf', 'hwdownload,format=bgr0',
                       '-frames', '1', '-update', '1',
                       filename], check=False, print_=False)

            if out.returncode == 0:
                captured = True

        if not captured:
            raise Exception('Could not capture screenshot within timeout.')

        self.screenshot_num += 1
        return filename


class Mouse():
    def __init__(self):
        run(['modprobe', 'uinput'], check=True)

        # Maximum coordinate for both axis
        self.ABS_MAX = 2**16

        # initialize the uinput device
        self.device = uinput.Device((
            uinput.ABS_X + (0, self.ABS_MAX, 0, 0),
            uinput.ABS_Y + (0, self.ABS_MAX, 0, 0),
            uinput.BTN_LEFT,
            uinput.BTN_MIDDLE,
            uinput.BTN_RIGHT,
            uinput.REL_WHEEL,
        ))

        self.CLICK_HOLD_TIME = 0.1

    def move(self, x: float, y: float):
        """Moves the mouse cursor to specified absolute coordinate."""

        logger.info(f'Moving mouse to {x, y})')

        for uinput_axis, value in [(uinput.ABS_X, x), (uinput.ABS_Y, y)]:
            # Check if value between 0 and 1
            assert ((value >= 0) and (value <= 1))
            converted = int(value * self.ABS_MAX)
            self.device.emit(uinput_axis, converted, syn=False)

        # Both axis move at once
        self.device.syn()

    def click(self, button: str = 'left'):
        """Clicks the any button of the mouse.
        :param button: mouse button to click, defaults to 'left'
        Possible values 'left', 'right' or 'middle'.
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
        sleep(self.CLICK_HOLD_TIME)
        # release the button
        self.device.emit(uinput_button, 0)


class KB():
    """Wrapper class for keyboard library."""

    def __init__(self, wait_time=5):
        self.WAIT_TIME = wait_time

        def kb_decorator(fn):
            def wrapper(*args, **kwargs):
                # Format the arguments for logging
                kwargs_list = ["=".join((key, repr(value)))
                               for key, value in kwargs.items()]
                args_list = [repr(value) for value in list(args)]
                all_args = ", ".join(args_list + kwargs_list)
                logger.info(f'Calling keyboard.{fn.__name__}({all_args})')
                fn(*args, **kwargs)
                sleep(self.WAIT_TIME)
            return wrapper

        # Workarounds for keyboard library
        # keyboard.write types nothing if the delay is not set
        self.write = functools.partial(kb_decorator(keyboard.write), delay=0.1)
        self.send = kb_decorator(keyboard.send)
        # For some reason the first keypress is never sent
        # So this effectively does nothing
        keyboard.send('enter')


class GUI():
    def __init__(self, wait_time=5):
        self.WAIT_TIME = wait_time
        self.GDM_INIT_TIME = 10
        # Create the directory for screenshots
        self.screenshot_directory = '/tmp/SC-tests/' + str(int(time()))
        os.makedirs(self.screenshot_directory, exist_ok=True)

        self.mouse = Mouse()
        self.kb = KB(self.WAIT_TIME)

    def __enter__(self):
        self.screen = Screen(self.screenshot_directory)
        # By restarting gdm, the system gets into defined state
        run(['systemctl', 'restart', 'gdm'], check=True)
        # Cannot screenshot before gdm starts displaying
        # This would break the display
        sleep(self.GDM_INIT_TIME)

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
                              dsize=None,
                              fx=UPSCALING_FACTOR,
                              fy=UPSCALING_FACTOR,
                              interpolation=cv2.INTER_LANCZOS4)
        _, binary = cv2.threshold(upscaled, 120, 255, cv2.THRESH_BINARY_INV)
        image_data_str = pytesseract.image_to_data(binary)
        df = pd.read_csv(StringIO(image_data_str),
                         sep='\t', lineterminator='\n')

        yres, xres = binary.shape[:2]
        df[['left', 'width']] /= xres
        df[['top', 'height']] /= yres

        logger.debug(df)
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
        """Clicks matching word on the screen.

        """
        logger.info(f"Trying to find key='{key}' to click on.")

        end_time = time() + timeout
        item = None
        first_scr = None

        # Repeat screenshotting, until the key is found
        while time() < end_time:
            # Capture the screenshot
            screenshot = self.screen.screenshot()

            last_scr = screenshot
            if first_scr is None:
                first_scr = screenshot

            df = self._image_to_data(screenshot)
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
                break

        if item is None:
            raise Exception(f"Found no key='{key}' in screenshots " +
                            f"{first_scr} to {last_scr}")

        x = float(item['left'] + item['width']/2)
        y = float(item['top'] + item['height']/2)

        self.mouse.move(x, y)
        sleep(0.5)
        self.mouse.click()
        sleep(self.WAIT_TIME)

    def assert_text(self, key: str, timeout: float = 0):
        """
        Given key must be found in a screenshot before the timeout.

        If the key is not found, exception is raised.
        Zero timeout means that only one screenshot
        will be taken and evaluated.
        """

        logger.info(f"Trying to find key='{key}'")

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
        logger.info(f"Trying to find key='{key}'" +
                    " (it should not be in the screenshot)")

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
                raise Exception(f"The key='{key}' was found " +
                                f"in the screenshot {screenshot}")
