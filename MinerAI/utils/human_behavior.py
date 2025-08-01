# utils/human_behavior.py
import time
import random

def human_delay():
    time.sleep(random.uniform(0.7, 3.0))

def random_error_chance(prob=0.02):
    return random.random() < prob