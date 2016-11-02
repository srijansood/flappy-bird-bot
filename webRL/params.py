from os import path

def abs_path(file_path):
    """
    Path Wrapper
    """
    return path.join(path.dirname(__file__), file_path)

# Location of Chromedriver
chromedriver = "res/chromedriver"

# Location of
adblock_path = abs_path("res/adblock.crx")

# Game Related Params
game_url = "http://flappybird.io/"
elem_id = "testCanvas"
goal = abs_path("res/goal_images/terminal.png")
max_steps = 10000