import unittest
from dataclasses import dataclass
from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)

class TestPrepareVectors(unittest.TestCase):

    def test_init(self):
        #self.assertEqual(self.cache.__class__.__name__, "Cache")

    def test_vectors(self):
        vec1 = vec2d("example1")
        self.cache.set("item1", vec1)
        vec2 = vec2d("example2")
        self.cache.set("item2", vec2)
        val = self.cache.get("item1")
        self.assertEqual(val, vec2d("example1"))
