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
    cache.deleteCache(".*")

    def test_init(self):
        self.assertEqual(self.cache.__class__.__name__, "Cache")

    def test_insert(self):
        vec1 = vec2d("example1")
        self.cache.set("item1", vec1)
        vec2 = vec2d("example2")
        self.cache.set("item2", vec2)
        val = self.cache.get("item1")
        self.assertEqual(val, vec2d("example1"))
        val = self.cache.get("item2")
        self.assertEqual(val, vec2d("example2"))
        list = self.cache.listKeys(".*")
        self.assertEqual(list[0], "defaultCache:item1")
        self.assertEqual(list[1], "defaultCache:item2")
        list = self.cache.listKeysVals(".*")
        self.assertEqual(list[0][0], "defaultCache:item1")
        self.assertEqual(list[0][1], vec2d("example1"))
        self.assertEqual(list[1][0], "defaultCache:item2")
        self.assertEqual(list[1][1], vec2d("example2"))

    def test_insert_with_prefix(self):
        
        cache = Cache("prefix_insert_test", prefix="test")
        cache.deleteCache
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

    def test_arrangeKey(self):
        cache = Cache("test") # デフォルトprefixが無いケース
        key0 = "item"
        key = cache.arrangeKey(key0)
        self.assertEqual(key, f"defaultCache:{key0}") # 何の指定も無ければdefaultCacheがつく
        key = cache.arrangeKey(key0, mode="emb")
        self.assertEqual(key, f"defaultCache:mode=emb:{key0}") # modeがあれば追加
        key = cache.arrangeKey(key0, prefix="dev")
        self.assertEqual(key, f"dev:{key0}") # prefixがあれば追加
        key = cache.arrangeKey(key0, mode="emb", prefix="dev")
        self.assertEqual(key, f"dev:mode=emb:{key0}") # modeとprefixがあれば追加
        cache = Cache("test", prefix="stage") # デフォルトprefixがあるケース
        key = cache.arrangeKey(key0)
        self.assertEqual(key, f"stage:{key0}") # 何の指定も無ければdefaultがつく
        key = cache.arrangeKey(key0, mode="emb")
        self.assertEqual(key, f"stage:mode=emb:{key0}") # 何の指定も無ければdefaultがつく
        key = cache.arrangeKey(key0, prefix="dev")
        self.assertEqual(key, f"dev:{key0}") # prefix指定は上書き
        key = cache.arrangeKey(key0, mode="emb", prefix="dev")
        self.assertEqual(key, f"dev:mode=emb:{key0}") # prefixで上書きされてmodeもつくケース

        key0 = torch.tensor([1, 2])
        print(key0)
        key = cache.arrangeKey(key0, mode="emb")
        print(key)
        print(key.__class__)
        
        
if __name__ == '__main__':
    unittest.main()
