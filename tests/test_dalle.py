import unittest
import torch

from sp2i.cli import generate_images


class TestClass(unittest.TestCase):

    """Test case docstring."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cli(self):
        text = [["blue bear"]]
        generate_images(text)
