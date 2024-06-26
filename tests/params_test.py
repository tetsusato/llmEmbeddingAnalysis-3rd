import unittest
from dataclasses import dataclass
from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

from params import Params

logger = logging.getLogger(__name__)

class TestParamsModule(unittest.TestCase):

    params = Params("config.toml")
    print(params)
    print(params.config)

    def test_init(self):
        self.assertEqual(self.params.__class__.__name__, "Params")
        params = Params("config.toml")
        print(params)
        print(params.config)
        root = params.config["cache"]["root"]
        self.assertEqual(root, "defaultCache")

        params = Params() # デフォルト値使用
        print(params)
        print(params.config)
        root = params.config["cache"]["root"]
        self.assertEqual(root, "defaultCache")


