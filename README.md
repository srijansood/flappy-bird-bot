# flappy-bird-bot
Flappy Bird using Deep Reinforcement Learning

Options for (Visual) Goal Representation: 
  * Bird between Pillars (reward = inverse of distance) ([Template Matching] (http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html))
  * Score Change
  * Death Screen (negative reward)
  

Instructions:
 1. Install requirements in requirements.txt (pip install -r requirements.txt)
 2. Download [Chromedriver](http://chromedriver.storage.googleapis.com/index.html?path=2.24/) and save as chromedriver in webRL/res
 3. python DeepLearningFlappyBird/deep_q_network.py 
