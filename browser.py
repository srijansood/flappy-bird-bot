import os
import time
from PIL import Image
from io import BytesIO

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

"""
Requirements (apart from requirements.txt) -
chromedriver (add to PATH)- http://chromedriver.storage.googleapis.com/index.html?path=2.24/
"""

# --- Globals ---
game_url = "http://flappybird.io/"



def screenshot(browser, element, save=False):
    elem_location = element.location
    elem_size = element.size

    img = Image.open(BytesIO(browser.get_screenshot_as_png()))
    left = elem_location['x']
    top = elem_location['y']
    right = elem_location['x'] + elem_size['width']
    bottom = elem_location['y'] + elem_size['height']
    box = (left, top, right, bottom)
    # (29.0, 123.0, 509.0, 763.0)
    # Does not work, need to double

    img = img.crop((58, 246, 1018, 1526))

    if save:
        im_path = os.path.join(os.getcwd(), "images", str(int(time.time())) + '.png')
        img.save(im_path, 'PNG')
    return img


def get_game(game_url):

    # options = webdriver.Firefox()
    # optio
    # options.add_extension(os.path.join(os.getcwd(), "adblock.crx"))
    # browser = webdriver.Chrome(chrome_options=options)
    # options = Options()
    firefox_binary = FirefoxBinary("/Applications/Firefox.app/Contents/MacOS/firefox-bin")
    # caps = DesiredCapabilities.FIREFOX
    # caps["binary"] = "/Applications/FirefoxDeveloperEdition.app/Contents/MacOS/firefox"
    # caps["marionette"] = True
    browser = webdriver.Firefox(firefox_binary=firefox_binary)
    browser.get(game_url)

    game = WebDriverWait(browser, 15).until(
        EC.presence_of_element_located((By.ID, "testCanvas"))
    )

    return browser, game