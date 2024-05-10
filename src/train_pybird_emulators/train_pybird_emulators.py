#!/usr/bin/env python

# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

import random


def say_hello():
    """this is a doc string and will show up in the documentation when
    you run 'make docs'
    """
    greetings = ["hi ho!", "how do you do?", "what a nice day!"]
    print(random.choice(greetings))
