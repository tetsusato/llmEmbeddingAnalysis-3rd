import diskcache
import logging
from logging import config
import re
import sys
import time
import torch.multiprocessing as multiprocessing
from tqdm import tqdm
from params import Params

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
                 #, source_class
                 cache_filename,
                 prefix = None,
                 cache_enable = True):
        """
        Args:
        #    source_class: A class of items which is stored in a cache.
        #                  The class should be dataclass and define
        #                  a __repr__() method.
            cache_filename: A name of the file on the local storage
                            where the item is stored.
 
        """
        if prefix is not None:
            self.key_prefix = prefix
        else:
            self.key_prefix = "defaultCache"
        logger.debug(f"Creating a cache object. key_prefix={self.key_prefix}, saves to {cache_filename}")
        if cache_filename != "" and cache_filename is not None and cache_enable is not False:
            filename = f"CacheStorage/" \
                       + f"{self.key_prefix}/" \
                       + f"{cache_filename}"
            logger.debug(f"Cache file = {filename}")
            self.is_enable = True
            #self.db = diskcache.Cache(cache_filename)
            self.db = diskcache.Cache(filename)
        else:
            self.is_enable = False
            self.db = None

    def close(self):
        self.db.close()

    def getCacheKey(self, keyName,
                  file = None,
                  embFunc = None,
                  simFunc = None,
                  numTokens = None,
                  postfunc = None,
                  list1 = None,
                  list2 = None,
                  comment = None):
                    
        """
        keyName: 識別するための名前．基本はrequired
        targetFile: ドキュメントのファイル名．simFuncがNoneのときはrequired
        numTokens: ドキュメント先頭からのトークン数．postfuncがNoneならrequired
        getEmbFunc: 埋め込みベクトル計算関数．postfuncがNoneならrequired
        simFunc: 類似度計算関数

        postfunc: simFuncのための前処理関数
        hash1: simFuncに渡す引数のハッシュ --> hashはkey作成のためにしか使われていないからlist1でいい
        hash2: simFuncに渡す引数のハッシュ

        """
        key = ""
        #if keyName == "emb" and embFunc == self.getHeadPersistenceDiagramEmbeddings:
        #    raise ValueError("use getPersistenceDiagramEmbeddings")

        #if keyName == "dis" and simFunc == self.BottleneckSim:
        #    raise ValueError("use Bottleneck")

        if keyName == "emb":
            key += f"{keyName}"
            key += f":file={file.name}"
            key += f":embFunc={embFunc.__name__}"
            key += f":tokens={numTokens}"
        elif keyName == "embmat":
            key += f"{keyName}"            
            key += f":embFunc={embFunc.__name__}"
            key += f":simFunc={simFunc.__name__}"
            key += f":tokens={numTokens}"
        elif keyName == "simmat":
            key += f"{keyName}"
            key += f":embFunc={embFunc.__name__}"
            key += f":simFunc={simFunc.__name__}"
            key += f":tokens={numTokens}"
        elif keyName == "dis":
            key += f"{keyName}"
            key += f":postfunc={postfunc.__name__}"
            key += f":simFunc={simFunc.__name__}"
            hash1 = self.hash_algorithm(list1.tobytes()).hexdigest()
            key += f":hash1={hash1}"
            hash2 = self.hash_algorithm(list2.tobytes()).hexdigest()
            key += f":hash2={hash2}"
        if comment is not None:
            key += f":comment={comment}"
        return key

    # prefixがclassの識別子で，modeがclassの中のmethodやdataの識別子？
    def set(self, key, val, mode=None, tag=None, prefix=None):
        key = self.arrangeKey(key, mode=mode, tag=tag, prefix=prefix)                
                
        #key = f"{self.key_prefix}:{key}"
        logger.debug(f"key={key}")
        logger.debug(f"val={val}")
        #self.db[key] = val
        self.db.set(key, val, tag=tag)
        #self.db.sync()

    def arrangeKey(self, key, mode=None, tag=None, prefix=None):
        """
        Keyの修飾を計算して返す
        """
        if self.key_prefix is not None: # コンストラクタのprefixがデフォルトで使われる
            if prefix is None: # prefixが無ければ文句なしにデフォルト
                if mode is None:
                    key = f"{self.key_prefix}:{key}" # 特に指定がなければこの形式
                else:
                    key = f"{self.key_prefix}:mode={mode}:{key}" # modeがあれば追加
            else: # prefixがあればこっちが優先される
                if mode is None:
                    key = f"{prefix}:{key}" # 特に指定がなければこの形式
                else:
                    key = f"{prefix}:mode={mode}:{key}" # modeがあれば追加
        elif prefix is not None: # でおフォルトがなくてprefixの指定があれば使われる
            if mode is None:
                key = f"{prefix}:{key}" # modeも無ければkeyそのもの
            else:
                key = f"{prefix}:mode={mode}:{key}" # modeがあれば追加
        else: # prefix指定が一切無い場合
            if mode is None:
                key = f"{key}" # modeも無ければkeyそのもの
            else:
                key = f"mode={mode}:{key}" # modeがあれば追加
        return key
            
                
    def get(self, key, mode=None, tag=None, prefix=None, rawmode=False):
        if rawmode is False:
            key = self.arrangeKey(key, mode=mode, tag=tag, prefix=prefix)
        val = self.db.get(key, tag=tag)
        logger.debug(f"key={key}")
        logger.debug(f"val={val}")
        return val

    def listKeys(self, regexp=".*", mode=None, tag=None, prefix=None):
        key = regexp.replace("[", "\\[")
        key = key.replace("]", "\\]")
        """
        Return list of keys as an array.
        Args:
            regexp: (Optional) The key that matches regexp is returned.
        """
        iter = self.db.iterkeys(reverse=False)
        key = self.arrangeKey(key, mode=mode, tag=tag, prefix=prefix)
        
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

    def listKeysVals(self, regexp = ".*", mode=None, tag=None, prefix=None):
        """
        Return list of keys as an array.
        Args:
            regexp: (Optional) The key that matches regexp is returned.
        """
        keyList = self.listKeys(regexp, mode=mode, tag=tag, prefix=prefix)
        keyValList = [(key, self.get(key, rawmode=True)) for key in keyList]
        return keyValList
        
    def deleteCache(self, regexp = None, mode=None, tag=None, prefix=None):
        """
        return: Counter indicating the number of deleted items
        """
        
        keys = self.listKeys(regexp, mode=mode, tag=tag, prefix=prefix)

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
