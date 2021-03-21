import unittest

from sp2i.cli import generate_images


class TestClass(unittest.TestCase):

    """Test case docstring."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cli(self):
        generate_images("blue bear")
