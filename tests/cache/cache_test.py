import unittest
from cache.cache import Cache
from dataclasses import dataclass
import torch
from hydra import compose, initialize

from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)

@dataclass
class vec2d:
    name: str
    dim: int = 2

    def __repr__(self):
        #return f"{self.__class__.__name__}(name, dim)"
        return f"{self.__class__.__name__}(name={self.name}, dim={self.dim})"

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name}, dim={self.dim})"


class TestCacheModule(unittest.TestCase):
    with initialize(version_base=None, config_path="../../conf", job_name=__file__):
        cfg = compose(config_name="config.yaml", overrides=["+io=qwen2"], return_hydra_config=True)
    
    vec = vec2d("vec2d example")
    filename = "test"
    #cache = Cache(vec, filename)
    cache = Cache(cfg, filename)
    cache.deleteCache(".*")

    def test_init(self):
        self.assertEqual(self.cache.__class__.__name__, "Cache")

    def test_insert(self):
        self.cache.deleteCache(".*")
        vec1 = vec2d("example1")
        self.cache.set("item1", vec1)
        vec2 = vec2d("example2")
        self.cache.set("item2", vec2)
        val = self.cache.get("item1")
        self.assertEqual(val, vec2d("example1"))
        val = self.cache.get("item2")
        self.assertEqual(val, vec2d("example2"))
        list = self.cache.listKeys(".*")
        print(f"list={list}")
        self.assertEqual(list[0], "item1")
        self.assertEqual(list[1], "item2")
        list = self.cache.listKeysVals(".*")
        self.assertEqual(list[0][0], "item1")
        self.assertEqual(list[0][1], vec2d("example1"))
        self.assertEqual(list[1][0], "item2")
        self.assertEqual(list[1][1], vec2d("example2"))

    def test_insert_with_prefix(self):
        
        cache = Cache(self.cfg, "prefix_insert_test", prefix="test")
        cache.deleteCache
        # access time prefixing
        vec1 = vec2d("example1")
        cache.set("item1", vec1)
        vec2 = vec2d("example2")
        cache.set("item2", vec2)
        vec3 = vec2d("example3")
        cache.set("item3", vec3, tag = "special")
        list = cache.listKeys(".*")
        logger.info(f"list={list}")
        self.assertEqual(list[0], "item1")
        self.assertEqual(list[1], "item2")
        self.assertEqual(list[2], f"item3:tag=special") 

        self.assertEqual(cache.get("item1"), vec1)
        self.assertEqual(cache.get("item2"), vec2)
        self.assertEqual(cache.get("item3", tag="special"), (vec3, "special"))

        # key includes tensor
        key0 = torch.tensor([1, 2])
        vec4 = vec2d("example4")
        print(key0)
        #key = cache.arrangeKey(key0, mode="emb")
        cache.set(key0, vec4)

        list = cache.listKeys(".*")
        logger.info(f"list={list}")
        self.assertEqual(list[0], "item1")
        self.assertEqual(list[1], "item2")
        print(list[2])
        print(list[2].__class__)
        print(isinstance(list[2], str))                

    def test_arrangeKey(self):
        cache = Cache(self.cfg, "test") # デフォルトprefixが無いケース
        key0 = "item"
        key = cache.arrangeKey(key0)
        self.assertEqual(key, f"{key0}") # 何の指定も無ければそのまま
        key = cache.arrangeKey(key0, tag="test")
        self.assertEqual(key, f"{key0}:tag=test") # tagがあればこうなる

    def test_get(self):
        cache = Cache(self.cfg, "test") # デフォルトprefixが無いケース
        key0 = "item"
        key1 = cache.arrangeKey(key0)
        val = vec2d("example")
        self.assertEqual(key1, f"{key0}") # 何の指定も無ければそのまま
        key2 = cache.arrangeKey(key0, tag="test")
        self.assertEqual(key2, f"{key0}:tag=test") # tagがあればこうなる
        cache.set(key1, val)
        val1 = cache.get(key1)
        val2 = cache.get("item_none")
        self.assertEqual(val1, val)
        self.assertEqual(val2, None)

        
        
if __name__ == '__main__':
    unittest.main()
