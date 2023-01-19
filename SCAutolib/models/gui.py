from SCAutolib import run

import os
from time import sleep, time
import uinput
import cv2
import numpy as np
import pandas as pd
from io import StringIO
import pytesseract
import keyboard
import uinput

class Screen():
    '''Captures the screenshots'''

    def __init__(self, directory):
        self.directory=directory
        self.screenshot_num = 1

    def screenshot(self):
        '''
        Runs ffmpeg to take a screenshot.

        The filename of screenshot is then returned.
        If no screenshot could be captured, None object is returned.
        '''

        filename = f'/tmp/SC-tests/{self.screenshot_num}.png'
        out = run(['ffmpeg', '-y', '-f', 'kmsgrab', '-i', '-',
             '-vf', 'hwdownload,format=bgr0', '-frames', '1',
             filename
        ])

        if out.returncode != 0:
            filename = None

        return filename

class Mouse():
    def __init__(self):
        run(['modprobe', 'uinput'], check=True)
        sleep(5) # TODO remove later
        # initialize the uinput device
        self.device = uinput.Device([
            uinput.REL_X,
            uinput.REL_Y,
            uinput.BTN_LEFT,
            uinput.BTN_MIDDLE,
            uinput.BTN_RIGHT,
            uinput.REL_WHEEL,
        ])

    def move(self, x, y):
        '''Moves the mouse cursor to specified absolute coordinate.'''

        # Go to upper left corner
        self.device.emit(uinput.REL_X, -10000)
        sleep(0.1)
        self.device.emit(uinput.REL_Y, -10000)
        sleep(0.1)

        # Move to correct X coordinate
        for i in range(x):
            self.device.emit(uinput.REL_X, 1)
            sleep(0.01)

        # Move to correct Y coordinate
        for i in range(y):
            self.device.emit(uinput.REL_Y, 1)
            sleep(0.01)

    def click(self, button='left'):
        '''
        Clicks the any button of the mouse.

        Button value can be 'left', 'right' or 'middle'.
        '''

        button_map = {
            'left': uinput.BTN_LEFT,
            'right': uinput.BTN_RIGHT,
            'middle': uinput.BTN_MIDDLE,
        }
        uinput_button = button_map[button]

        # press the button
        self.device.emit(uinput_button, 1)
        # wait a little
        sleep(0.1)
        # release the button
        self.device.emit(uinput_button, 0)

class GUI():
    def __init__(self):
        # Create the directory for screenshots
        sceenshot_directory = '/tmp/SC-tests'
        os.mkdirs('/tmp/SC-tests', exist_ok=True)

        self.screen = Screen(screenshot_directory)
        self.mouse = Mouse()

        # Workaround for keyboard library
        # For some reason the first keypress is never sent
        keyboard.send('enter')

        # By restarting gdm, the system gets into defined state
        run(['systemctl', 'restart', 'gdm'], check=True)

    def _image_to_data(self, filename):
        '''
        Convert screenshot into dataframe of words with their coordinates.
        '''
        image = cv2.imread(filename)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(grayscale, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        thresh, binary = cv2.threshold(grayscale, 120, 255, cv2.THRESH_BINARY_INV)
        image_data_str = pytesseract.image_to_data(binary)
        df = pd.read_csv(StringIO(image_data_str), sep='\t', lineterminator='\n')
        return df


    def _images_same(filename1, filename2):
        '''Compare two images, return True if they are completely identical.'''
        im1 = cv2.imread(filename1)
        im2 = cv2.imread(filename2)

        # Is the resolution the same
        if im1.shape != im2.shape:
            return False

        # Check if value of every pixel is the same
        if np.bitwise_xor(image1, image2).any():
            return False

        return True


    def keyboard_write(self, text):
        # delay is necesary as a workaround for keybord library
        keyboard.write(text, delay=0.1)


    def click_on(self, key: str, timeout=30):
        end_time = time() + timeout

        # Repeat screenshotting, until the key is found
        while time() < end_time:
            # Capture the screenshot
            screenshot = None
            while screenshot is None and time() < end_time:
                # If capturing fails, try again
                screenshot = self.screen.screenshot()

            df = self._image_to_data(filename)

            selection = df['text'] == key

            # If there is no matching word, try again
            if selection.sum() == 0:
                continue

            # Exactly one word matching, exit the loop
            elif selection.sum() == 1:
                item = df[selection]
                break

            # More than one word matches, choose the first match
            # Probably deterministic, but it should not be relied upon
            else:
                print('Too many to choose from') # TODO replace the print function
                item = df.iloc[0]

        if time() >= end_time:
            raise Exception('Found no matching key in any screenshot.')

        x = int(item['left'] + item['width']/2)
        y = int(item['top'] + item['height']/2)

        self.mouse.move(x, y)
        self.mouse.click()


    def wait_still_screen(self, time_still=5, timeout=60):
        '''
        Wait while until the screen content stops changing.

        When nothing on screen has changed for time_still seconds, continue.
        If the screen content is changing permanently,
        fail with exception after timeout seconds.
        '''

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
            if last_screenshot is None or not _images_same(screenshot, last_screenshot):
                last_screenshot = screenshot
                # Image has changed, refresh time_still_end
                time_still_end = time() + time_still

        # If the loop was ended by timeout, it is an error
        if time() >= timeout_end:
            raise Exception('Screen contents were changing until timeout.')

















