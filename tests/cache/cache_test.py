import unittest
from cache.cache import Cache
from dataclasses import dataclass
import torch
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
    vec = vec2d("vec2d example")
    filename = "test"
    #cache = Cache(vec, filename)
    cache = Cache(filename)
    cache.delete(".*")
    cache.clear()

    def test_init(self):
        self.assertEqual(self.cache.__class__.__name__, "Cache")

    def test_clear(self):
        self.assertEqual(self.cache.__class__.__name__, "Cache")
        self.cache.clear()
        
    def test_insert(self):
        vec1 = vec2d("example1")
        self.cache.set("item1", vec1)
        vec2 = vec2d("example2")
        self.cache.set("item2", vec2)
        val = self.cache.get("item1")
        self.assertEqual(val, vec2d("example1"))
        val = self.cache.get("item2")
        self.assertEqual(val, vec2d("example2"))
        list = self.cache.listKeys(".*", returnFullPath=True)
        self.assertEqual(len(list), 2)
        self.assertEqual(list[0], "//defaultCache/item1")
        self.assertEqual(list[1], "//defaultCache/item2")
        list = self.cache.listKeysVals(".*", returnFullPath=True)
        self.assertEqual(list[0][0], "//defaultCache/item1")
        self.assertEqual(list[0][1], vec2d("example1"))
        self.assertEqual(list[1][0], "//defaultCache/item2")
        self.assertEqual(list[1][1], vec2d("example2"))
        list = self.cache.listKeys(".*")
        self.assertEqual(len(list), 2)
        self.assertEqual(list[0], "item1")
        self.assertEqual(list[1], "item2")
        list = self.cache.listKeysVals(".*")
        self.assertEqual(list[0][0], "item1")
        self.assertEqual(list[0][1], vec2d("example1"))
        self.assertEqual(list[1][0], "item2")
        self.assertEqual(list[1][1], vec2d("example2"))

        count = self.cache.delete("item1")
        self.assertEqual(count, 1)
        val = self.cache.get("item1")
        self.assertEqual(val, None)
        list = self.cache.listKeys(".*")
        self.assertEqual(len(list), 1)
        self.assertEqual(list[0], "item2")
        
    """
    def test_insert_with_prefix(self):
        
        cache = Cache("prefix_insert_test", prefix="test")
        cache.delete
        # access time prefixing
        vec1 = vec2d("example1")
        cache.set("item1", vec1)
        vec2 = vec2d("example2")
        cache.set("item2", vec2)
        vec3 = vec2d("example3")
        cache.set("item1", vec3, prefix = "special")
        vec4 = vec2d("example4")
        cache.set("item4emb", vec4, mode="emb")
        list = cache.listKeys(".*")
        logger.info(f"list={list}")
        self.assertEqual(list[0], "test:item1")
        self.assertEqual(list[1], "test:item2")
        self.assertEqual(list[2], f"test:mode=emb:item4emb") 

        self.assertEqual(cache.get("item1"), vec1)
        self.assertEqual(cache.get("item2"), vec2)
        self.assertEqual(cache.get("item1", prefix="special"), vec3)
        self.assertEqual(cache.get("item4emb", mode="emb"), vec4)

        # construntor prefixing
        cache = Cache("prefix_insert_test_2", prefix="very special")
        vec1 = vec2d("example1")
        cache.set("item1", vec1)
        vec2 = vec2d("example2")
        cache.set("item2", vec2)
        vec3 = vec2d("example3")
        cache.set("item1", vec3, prefix="special") # overwrite
        list = cache.listKeys(".*")
        logger.info(f"list={list}")
        self.assertEqual(list[0], "very special:item1")
        self.assertEqual(list[1], "very special:item2")

        # key includes tensor
        key0 = torch.tensor([1, 2])
        vec4 = vec2d("example4")
        print(key0)
        #key = cache.arrangeKey(key0, mode="emb")
        cache.set(key0, vec4, mode="emb")

        list = cache.listKeys(".*")
        logger.info(f"list={list}")
        self.assertEqual(list[0], "very special:item1")
        self.assertEqual(list[1], "very special:item2")
        print(list[2])
        print(list[2].__class__)
        print(isinstance(list[2], str))                
    """

    def test_arrangeKey(self):
        cache = Cache("test") # デフォルトprefixが無いケース
        key0 = "item"
        key = cache.arrangeKey(key0)
        self.assertEqual(key, f"//defaultCache/{key0}") # 何の指定も無ければdefaultCacheがつく
        key = cache.arrangeKey(key0, prefix="dev")
        self.assertEqual(key, f"//dev/{key0}") # prefixがあれば追加
        cache = Cache("test", prefix="stage") # デフォルトprefixがあるケース
        key = cache.arrangeKey(key0)
        self.assertEqual(key, f"//stage/{key0}") # 何の指定も無ければdefaultがつく
        key = cache.arrangeKey(key0, tag="all", prefix="dev")
        self.assertEqual(key, f"/all/dev/{key0}") # tag/prefix指定


        
        
if __name__ == '__main__':
    unittest.main()
