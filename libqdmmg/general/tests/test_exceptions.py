'''

author: Linus Bjarne Dittmer

This class contains unittests to test the file

    libqdmmg/general/exceptions.py

'''

import unittest
import numpy
import libqdmmg.general as gen
import libqdmmg.simulate as sim

class TestInvalidIntegralRequestString(unittest.TestCase):
   
    def setUp(self):
        self.e = gen.IIRSException("example", "")

    def tearDown(self):
        del self.e

    def test_methods(self):
        self.assertTrue(hasattr(self.e, '__str__'))
        self.assertTrue(isinstance(self.e, Exception))


if __name__ == '__main__':
    unittest.main()

