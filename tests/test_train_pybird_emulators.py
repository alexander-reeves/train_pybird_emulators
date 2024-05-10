# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

import pkg_resources
import pytest

try:
    from train_pybird_emulators import train_pybird_emulators
except pkg_resources.DistributionNotFound:
    raise ImportError("did you run 'pip install -e .' for your project")


def test_import():
    train_pybird_emulators.say_hello()


"""
you are looking for setup / teardown methods? py.test has fixtures:
    http://doc.pytest.org/en/latest/fixture.html
you find examples below
"""


@pytest.fixture
def one():
    print("setup")
    yield 1
    print("teardown")


def test_something(one):
    assert one == 1
