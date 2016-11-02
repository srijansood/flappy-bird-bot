import os
import time
import params
from PIL import Image
from io import BytesIO

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC


def screenshot(browser, element, grayscale=True, save=False):
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

    if grayscale:
        img = img.convert('L')

    if save:
        im_path = os.path.join(os.getcwd(), "images", str(int(time.time())) + '.png')
        img.save(im_path, 'PNG')

    return img


def get_game(game_url=params.game_url):

    options = webdriver.ChromeOptions()
    options.add_extension(os.path.join(os.getcwd(), params.adblock_path))
    browser = webdriver.Chrome(params.chromedriver, chrome_options=options)
    browser.get(game_url)

    game = WebDriverWait(browser, 15).until(
        EC.presence_of_element_located((By.ID, params.elem_id))
    )

    actions = ActionChains(browser)
    actions.move_to_element_with_offset(game, 160, 375)  # move to 205, 592 (positition of RESTART)
    actions.perform()

    return browser, game
