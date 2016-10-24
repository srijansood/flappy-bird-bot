import cv2

import numpy as np
import os
import rewardFunction as rf
from PIL import Image
from browser import get_game, screenshot, ActionChains

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
        import ipdb; ipdb.set_trace()
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

        # Template Matching for Task Representation
        """
        res = cv2.matchTemplate(bmp,goal,cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        stateROI = bmp[top_left[1]:top_left[1] + w, top_left[0]:top_left[0] + h].copy()

        goalWidth, goalHeight = stateROI.shape[:2]
        goalROI = goal[0:goalWidth, 0:goalHeight].copy()

        stateROI = cv2.cvtColor(stateROI, cv2.COLOR_BGR2GRAY)
        goalROI = cv2.cvtColor(goalROI, cv2.COLOR_BGR2GRAY)

        reward = 1 / (np.exp((rf.calculateDistance(stateROI, goalROI))))
        """
        # for f in os.listdir(GOAL_TEMPLATES):
        for f in ['03.jpg']:
            # game_state_img = cv2.imread("/Users/Srijan/Dev/Research/flappy-bird-bot/images/between.png")
            output = game_state_img.copy()
            game_state_img = cv2.cvtColor(game_state_img, cv2.COLOR_BGR2GRAY)
            goal_img = cv2.imread(os.path.join(GOAL_TEMPLATES, f))
            goal_img = cv2.cvtColor(goal_img, cv2.COLOR_BGR2GRAY)

            res = cv2.matchTemplate(image=game_state_img, templ=goal_img, method=cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            h, w = goal_img.shape
            best_match = game_state_img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w].copy()
            dist1 = rf.calculateDistance(best_match, goal_img)
            dist2 = sum(sum(abs(goal_img - best_match)))
            rew1 = float(1) / float(np.exp(dist1))
            rew2 = float(1) / float(np.exp(dist2))

            print("Hog: {} <> Reward: {}   |   SSD: {} <> Reward: {}".format(dist1, rew1, dist2, rew2))
            # cv2.rectangle(output, top_left, (top_left[0] + w, top_left[1] + h), color)

            # Image.fromarray(output).show()



            # stateROI = game_state_img[top_left[1]:top_left[1] + w, top_left[0]:top_left[0] + h].copy()
            #
            # goalWidth, goalHeight = stateROI.shape[:2]
            # goalROI = goal_img[0:goalWidth, 0:goalHeight].copy()
            #
            # stateROI = cv2.cvtColor(stateROI, cv2.COLOR_BGR2GRAY)
            # goalROI = cv2.cvtColor(goalROI, cv2.COLOR_BGR2GRAY)



            # reward = 1 / (np.exp((rf.calculateDistance(stateROI, goalROI))))

        reward = rew1

        return game_state.astype(np.uint8), reward, terminal, self.score

    def screenshot(self):
        img, score = screenshot(self.browser, self.game)

        # (1280, 960, 3)
        return np.array(img), np.array(score)

        # # (1228800, 3)
        # return np.array(img.getdata())

        # # (960, 1280, 3)
        # return np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
