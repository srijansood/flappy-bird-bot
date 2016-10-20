import numpy as np
from browser import get_game, screenshot, ActionChains

from PIL import Image

"""
Defining wrapper game for online version
"""

LAMBDA = .3
SCORE_THRESHOLD = 0.5

class GameState:
    """
    Used to describe current state of the game
    """

    def __init__(self):
        self.steps = 0
        self.browser, self.game = get_game()
        self.state, self.score = self.screenshot()


    def frame_step(self, input_actions):
        """
        :param input_actions: "one hot" encoding of actions
        :return: next (averaged) state and reward (False and 0 for "terminal" and "score")
        """

        self.steps += 1

        # actions are a vector with a bit corresponding to action to take
        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1: # TODO: also serves as restart right now
            ActionChains(self.browser).click().perform()

        # exponential moving average of image frames
        game_state_img, score = self.screenshot()
        self.state = (1 - LAMBDA) * game_state_img + LAMBDA * self.state
        new_state_img = Image.fromarray(np.uint8(self.state))
        # new_state_ img.show()
        game_state = self.state.copy()

        terminal = False

        # Check if Score Area changed considerably for Reward
        # Alternatives - Template Matching for bird between pipes or for death screen
        manhattan_diff = sum(sum(abs(score - self.score)))
        print(manhattan_diff)
        self.score = score

        reward = 0

        return game_state.astype(np.uint8), reward, terminal, self.score

    def screenshot(self):
        img, score = screenshot(self.browser, self.game)

        # (1280, 960, 3)
        return np.array(img), np.array(score)

        # # (1228800, 3)
        # return np.array(img.getdata())

        # # (960, 1280, 3)
        # return np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
