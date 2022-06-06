import os

DIR = os.path.dirname(os.path.realpath(__file__))

# make a cache folder if it doesn't exist
if not os.path.exists(f"{DIR}/cache"):
    os.makedirs(f"{DIR}/cache")
