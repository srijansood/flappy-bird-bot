from browser import get_game, screenshot
from random import randint
from time import sleep

from ipdb import set_trace as pdb


actions = [" ", None]  # List of permissible actions


def main(browser, game):
    for i in xrange(100):
        # screen = screenshot(browser, game)

        action = actions[randint(0, 1)]
        if action:
            # game.send_keys(action)
            game.click()
            print("click")
        else:
            sleep(0.3)
            print("wait")
    browser.quit()

if __name__ == "__main__":
    game_url = "http://flappybird.io/"
    browser, game = get_game(game_url)
    # pdb()
    main(browser, game)