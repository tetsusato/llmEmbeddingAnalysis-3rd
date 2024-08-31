import diskcache
import logging
from logging import config
import re
import sys
import time
import torch.multiprocessing as multiprocessing
from tqdm import tqdm
from params import Params
from omegaconf import DictConfig, OmegaConf

config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)


class Cache():
    """
        prefixで名前空間を分けることができる．
        prefixはCache("filename", prefix="...")で指定できるほか，
        set(key, val, prefix="...")で他のインスタンスへのアクセスもできる．
        filenameがNoneか""なら，キャッシュはオフとなる
        オフかどうかは，is_enableで判断してもらう
        というか，アクセスしてきてもキャッシュが無いって返事すれば良いのか
        いや，それだけだと，無かったら計算してキャッシュに入れるとか無駄な処理が走るか？
        キャッシュに入れるもsetが呼ばれるだけだからら，内部でチェックしてスルーでもいいかもn
    """

    key_prefix = None
    is_enable = True
    def __init__(self,
                 cfg: DictConfig,
                 #, source_class
                 cache_filename,
                 prefix = None,
                 reset = False
                ):
        """ __init__
        Args:
            cfg: Config object. (required)
            cache_filename: A name of the file on the local storage
                            where the item is stored. (required)
 
        """
        top_dir = cfg.cache.top_dir
        root = cfg.cache.root
        if prefix is not None:
            self.key_prefix = prefix
        else:
            #self.key_prefix = "defaultCache"
            self.key_prefix = root
        logger.debug(f"Creating a cache object. key_prefix={self.key_prefix}, saves to {cache_filename}, flag cache_enable={cfg.cache.enable}")
        if cfg.cache.enable:
            """
            filename = f"CacheStorage/" \
                       + f"{self.key_prefix}/" \
                       + f"{cache_filename}"
            """
            filename = f"{top_dir}/" \
                       + f"{self.key_prefix}/" \
                       + f"{cache_filename}"
            logger.debug(f"Cache file = {filename}")
            self.is_enable = True
            #self.db = diskcache.Cache(cache_filename)
            self.db = diskcache.Cache(filename)
            is_reset = cfg.cache.reset
            if reset:
                is_reset = reset
            if reset:
                logger.info(f"Cache will be cleared")
                count = self.db.clear()
                logger.debug(f"Removed {count} items.")
        else:
            self.is_enable = False
            self.db = None
        logger.debug(f"db={self.db}")

    def close(self):
        self.db.close()

    def getCacheKey(self, keyName,
                   ):
                    
        """
        keyName: 識別するための名前．基本はrequired
        """
        return key

    # prefixがclassの識別子で，modeがclassの中のmethodやdataの識別子？
    def set(self, key, val, tag=None):
        key = self.arrangeKey(key, tag=tag)
                
        #key = f"{self.key_prefix}:{key}"
        logger.debug(f"key={key}")
        logger.debug(f"val={val}")
        #self.db[key] = val
        self.db.set(key, val, tag=tag)
        #self.db.sync()

    def arrangeKey(self, key, tag=None):
        """
        Keyの修飾を計算して返す
        """
        if tag is not None:
            key = f"{key}:tag={tag}" # modeがあれば追加
        else:
            key = f"{key}"
        return key
            
                
    def get(self, key, tag=None):
        key = self.arrangeKey(key, tag=tag)
        val = self.db.get(key, tag=False)
        logger.debug(f"key={key}")
        logger.debug(f"val={val}")
        return val

    def listKeys(self, regexp=".*", tag=None):
        key = regexp.replace("[", "\\[")
        key = key.replace("]", "\\]")
        """
        Return list of keys as an array.
        Args:
            regexp: (Optional) The key that matches regexp is returned.
        """
        iter = self.db.iterkeys(reverse=False)
        key = self.arrangeKey(key, tag=tag)
        
        substring = f"{key}"

        hit_keys = []
        #logger.debug(f"target key={substring}")
        count = 0
        for key in iter:
            logger.debug(f"target record={key} class={type(key)}")
            logger.debug(f"target regexp={substring}")
            #print(re.search(substring, key))
            if isinstance(key, str):
                if re.search(substring, key):
                    logger.debug("match!")
                    hit_keys.append(key)
            else:
                logger.warn(f"Skip: key = {key}, class = {key.__class__}. Only supported class is string.")
                
            count += 1
        logger.debug(f"total number of items = {count}")
        return hit_keys

    def listKeysVals(self, regexp = ".*", tag=None):
        """
        Return list of keys as an array.
        Args:
            regexp: (Optional) The key that matches regexp is returned.
        """
        keyList = self.listKeys(regexp, tag=tag)
        keyValList = [(key, self.get(key)) for key in keyList]
        return keyValList
        
    def deleteCache(self, regexp = None, tag=None):
        """
        return: Counter indicating the number of deleted items
        """
        
        keys = self.listKeys(regexp, tag=tag)

        count = 0
        logger.info(f"delete target keys[0:5]={keys[0:5]}")
        #input("OK?")
        for key in keys:
            logger.debug(f"deleting candidate key={key}")
            #input("OK?")
            if self.db.delete(key):
                count += 1
        logger.info(f"deleted {count} records")
        return count

if __name__ == "__main__":
    logger.info(f"■■■■■■■■■■■■■■■■■■■■■■■ main")
    args = sys.argv
    print(args)
    if args[1] == "--list-cache":
        if args[2] == "rwkv":
            logger.info(f"list rwkv embedding cache")
            start = time.time()
            #filename = "rwkv_gutenberg"
            params = Params("config.toml")
            filename = params.cache_filename
            model_name = 'RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth'
            tokenizer_name = '20B_tokenizer.json'
            from rwkv_runner import rwkv
            r = rwkv(model_name, tokenizer_name, model_load = False)
            c = Cache(Cache, filename, model_name, r.tokenizer)
            keys = c.listKeys("emb", embFunc = r.getRwkvEmbeddings)

            #filename = "cache_test"
            #r = rwkv(model_name, tokenizer_name, model_load = False)
            #keys = c.listCache("all")
            logger.info(f"target keys[0:3]={keys[0:3]}")
            for key in keys:
                logger.info(f"hit: {key}")
        elif args[2] == "use":
            logger.info(f"list use embedding cache")
            start = time.time()
            #from rwkv_runner import rwkv
            #r = rwkv(model_name, tokenizer_name, model_load = False)
            #c = Cache(Cache, filename, model_name, r.tokenizer)
            #keys = c.listCache("emb", embFunc = r.getRwkvEmbeddings)

            #filename = "cache_test"
            #filename = "use_gutenberg"
            params = Params("config-use.toml")
            filename = params.cache_filename

            model_name = 'RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth'
            tokenizer_name = '20B_tokenizer.json'
            #r = rwkv(model_name, tokenizer_name, model_load = False)
            from use_wrapper import use_wrapper
            u = use_wrapper(model_name, tokenizer_name, model_load = False)
            c = Cache(Cache, filename, model_name, u.tokenizer)
            keys = c.listKeys("emb", embFunc = u.getUseEmbeddings)
            #keys = c.listCache("all")
            logger.info(f"target keys[0:3]={keys[0:3]}")
            for key in keys:
                logger.info(f"hit: {key}")

        elif args[2] == "pd":
            logger.info(f"list PD embedding cache")
            start = time.time()
            #from rwkv_runner import rwkv
            #r = rwkv(model_name, tokenizer_name, model_load = False)
            #c = Cache(Cache, filename, model_name, r.tokenizer)
            #keys = c.listCache("emb", embFunc = r.getRwkvEmbeddings)

            #filename = "cache_test"
            #filename = "use_gutenberg"
            params = Params("config-use.toml")
            filename = params.cache_filename

            model_name = 'RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth'
            tokenizer_name = '20B_tokenizer.json'
            #r = rwkv(model_name, tokenizer_name, model_load = False)
            from use_wrapper import use_wrapper
            u = use_wrapper(model_name, tokenizer_name, model_load = False)
            c = Cache(Cache, filename, model_name, u.tokenizer)
            keys = c.listKeys("emb", embFunc = u.getPersistenceDiagramEmbeddings)
            #keys = c.listCache("all")
            logger.info(f"target keys[0:3]={keys[0:3]}")
            for key in keys:
                logger.info(f"hit: {key}")
