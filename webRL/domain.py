from collections import deque

import numpy as np
import os
import params
from PIL import Image
from browser import get_game, screenshot, ActionChains
from sklearn.metrics import mean_squared_error

"""
Defining wrapper game for online version
"""

LAMBDA = .3
SCORE_THRESHOLD = 0.5
GOAL_TEMPLATES = os.path.join(os.getcwd(), "goal_images/aligned")


class GameState:
    """
    Used to describe current state of the game
    """

    def __init__(self):
        self.steps = 0
        self.browser, self.game = get_game()
        self.states = deque(4*[self.screenshot()[0]], maxlen=4)    # stacked frames
        self.goal = np.array(Image.open(params.goal).convert('L'))

    def frame_step(self, input_actions):
        """
        :param input_actions: "one hot" encoding of actions
        :return: next (averaged) state and reward (False and 0 for "terminal" and "score")
        """

        self.steps += 1

        # actions are a vector with a bit corresponding to action to take
        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:   # Also serves as game restart
            ActionChains(self.browser).click().perform()

        game_state_img, game_state_img_grayscale = self.screenshot()
        self.states.popleft()   # remove oldest frame
        self.states.append(game_state_img)

        reward = get_reward(game_state_img_grayscale, goal_img=self.goal)
        terminal = self.steps == params.max_steps

        return np.asarray(self.states, dtype=np.uint8), reward, terminal

    def screenshot(self):
        """
        :return: (img, 80x80 img) - Grayscale Screenshot of Game
        """
        img = screenshot(self.browser, self.game, grayscale=False)

        # (1280, 960, 3)
        return np.array(img), np.array(img.convert('L'))

        # # (1228800, 3)
        # return np.array(img.getdata())

        # # (960, 1280, 3)
        # return np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)


def get_reward(state_img, goal_img):
    """
    Options:
        1. "Positive" Goal, use template matching on the state_img, return inverse of distance to goal_img
        2. "Negative" Goal, detect if state_img matches the death screen, return negative reward if so
    """
    threshold = 3.0
    dist = mean_squared_error(state_img, goal_img)
    print dist
    if dist < threshold:
        # death
        return -1
    else:
        return 0
