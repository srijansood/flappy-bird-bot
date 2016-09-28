import pyscreenshot as ScreenGrab
import os
import time
from PIL import Image
from pdb import set_trace as pdb

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

"""
Requirements (apart from reuirements.txt) -
chromedriver (add to PATH)- http://chromedriver.storage.googleapis.com/index.html?path=2.24/
"""

# --- Globals ---
game_url = "http://flappybird.io/"

# Retina MBP13, FB on left in Chrome
x_margin = 44
y_margin = 218
bounds = (x_margin + 1, y_margin + 1, x_margin + 481, y_margin + 641)

def screenGrab():
    im = ScreenGrab.grab(bounds)
    im_path = os.path.join(os.getcwd(), str(int(time.time())) +
    '.png')
    im.save(im_path, 'PNG')

    return im

def selenium_grab(browser):
    im_path = os.path.join(os.getcwd(), "images", str(int(time.time())) +
    '.png')
    browser.get_screenshot_as_file(im_path)
    return Image.open(im_path)

def web_driver():
    browser = webdriver.Chrome()
    browser.get(game_url)
    screenshot = selenium_grab(browser)

    # game = browser.find_element_by_id("testCanvas")
    #

    game = WebDriverWait(browser, 15).until(
        EC.presence_of_element_located((By.ID, "testCanvas"))
    )
    print("Found game")
    input("Continue")
    for i in xrange(100):
        print("Space")
        game.click()

    pdb()



def main():
    web_driver()

if __name__ == '__main__':
    main()
