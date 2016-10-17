import numpy as np
from browser import get_game, screenshot, ActionChains

"""
Defining wrapper game for online version
"""

LAMBDA = .3

class GameState:
    """
    Used to describe current state of the game
    """

    def __init__(self):
        self.steps = 0
        self.score = 0
        self.browser, self.game = get_game()
        self.state = self.screenshot()

    def frame_step(self, input_actions):
        """
        :param input_actions: "one hot" encoding of actions
        :return: next (averaged) state and reward (False and 0 for "terminal" and "score")
        """

        self.steps += 1

        # actions are a vector with a bit corresponding to action to take
        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            ActionChains(self.browser).click().perform()

        # exponential moving average of image frames
        game_state_img = self.screenshot()
        self.state = LAMBDA * game_state_img + (1 - LAMBDA) * (self.state)
        game_state = self.state.copy()

        terminal = False

        # TODO: Implement reward
        reward = 0

        return game_state.astype(np.uint8), reward, terminal, self.score

    def screenshot(self):
        img = screenshot(self.browser, self.game)
        return np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)