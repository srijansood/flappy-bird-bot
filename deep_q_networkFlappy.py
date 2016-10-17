#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 50000. # timesteps to observe before training
#EXPLORE = 1000000. # frames over which to anneal epsilon
EXPLORE = 100000
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 400000 # number of previous transitions to remember
BATCH = 100 # size of minibatch
FRAME_PER_ACTION = 1
K = 1 # only select an action every Kth frame, repeat prev for others
VARIABLES = 0
REP = int(sys.argv[1]) # image type
RUNS = 10000000

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # input layer
    s = tf.placeholder("float", [None, 80, 80, 1])

     ############################ Create train parameters  ############################
    W_conv1 = weight_variable([8, 8, 1, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([6400, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    trainParameters = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2]

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_pool3_flat = tf.reshape(h_conv3, [-1, 6400])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    ############################ Create target parameters ############################
    target_W_conv1 = weight_variable([8, 8, 1, 32])
    target_b_conv1 = bias_variable([32])

    target_W_conv2 = weight_variable([4, 4, 32, 64])
    target_b_conv2 = bias_variable([64])

    target_W_conv3 = weight_variable([3, 3, 64, 64])
    target_b_conv3 = bias_variable([64])

    target_W_fc1 = weight_variable([6400, 512])
    target_b_fc1 = bias_variable([512])

    target_W_fc2 = weight_variable([512, ACTIONS])
    target_b_fc2 = bias_variable([ACTIONS])

    targetParameters = [target_W_conv1, target_b_conv1, target_W_conv2, target_b_conv2, target_W_conv3, target_b_conv3, target_W_fc1, target_b_fc1, target_W_fc2]

    # hidden layers
    target_h_conv1 = tf.nn.relu(conv2d(s, target_W_conv1, 4) + target_b_conv1)
    target_h_conv2 = tf.nn.relu(conv2d(target_h_conv1, target_W_conv2, 2) + target_b_conv2)
    target_h_conv3 = tf.nn.relu(conv2d(target_h_conv2, target_W_conv3, 1) + target_b_conv3)
    target_h_pool3_flat = tf.reshape(target_h_conv3, [-1, 6400])
    target_h_fc1 = tf.nn.relu(tf.matmul(target_h_pool3_flat, target_W_fc1) + target_b_fc1)

    # readout layer
    target_readout = tf.matmul(target_h_fc1, target_W_fc2) + target_b_fc2

    return s, readout, target_readout, trainParameters, targetParameters

def trainNetwork(s, readout, target_readout, trainParameters, targetParameters, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing

    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')
    results_file = open("logs_" + GAME + "/" + str(REP) + "results.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal, _ = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    x_t = np.reshape(x_t, (80, 80, 1))
    s_t = x_t

    sess.run(tf.initialize_all_variables())

    epsilon = INITIAL_EPSILON
    episode = 0
    steps = 0
    totalReward = 0

    for t in range(0, RUNS):
        # update target network every C steps
        if t % 10000 == 0:
            for i in range(0, len(trainParameters)):
                sess.run(targetParameters[i].assign(trainParameters[i]))

        if t % 100 == 0:
            print("t: " + str(t))

        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0

        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(ACTIONS)
        else:
            action_index = np.argmax(readout_t)

        a_t[action_index] = 1

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        terminal = False

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal, _ = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = x_t1

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t >= OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = target_readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                # if terminal only equals reward
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch})

        # update the old values
        s_t = s_t1
        steps += 1

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if terminal and t > OBSERVE:
            if episode % 100 == 0:
                runPolicy(s, episode, results_file, state, epsilon, readout)
            print(episode)

            episode += 1
            steps = 0

def runPolicy(s, episode, results_file, state, epsilon, readout):
    #print("Testing")
    terminal = False
    testGame = game.GameState()
    steps = 0
    totalReward = 0

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal, newScore = testGame.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    x_t = np.reshape(x_t, (80, 80, 1))
    s_t = x_t
    actions = ""

    while not terminal:
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = np.argmax(readout_t)
        a_t[action_index] = 1

        # Run the selected action
        x_t1_colored, r_t, terminal, newScore = testGame.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t = x_t1

        totalReward += r_t
        steps = steps + 1

    print("EPISODE", episode, "/ STEPS", steps, "/ SCORE", newScore, "/ EPSILON", epsilon,  "/ REWARD", totalReward,  "/ Q_MAX %e" % np.max(readout_t))

    out = "{},{},{},{}\n".format(episode, steps, totalReward, newScore)
    out = out.replace(",,", ",")
    out = out.replace("(", "")
    out = out.replace(")", "")
    out = out.replace(" ", "")
    totalReward = 0
    results_file.write(out)
    results_file.flush()

def playGame():
    sess = tf.Session()

    with sess.as_default():
        s, readout, target_readout, trainParameters, targetParameters = createNetwork()
        trainNetwork(s, readout, target_readout, trainParameters, targetParameters, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
