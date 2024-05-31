import toml

import logging
from logging import config

config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)

class Params():
    def __init__(self, filename="config.toml"):
        """
        引数で与えられるファイル名のファイルからパラメータを読み込み，
        インスタンス変数configに内容をセットする
        ファイル名が省略された場合，デフォルト値config.tomlが使われる
        """
        self.filename = filename

        #config = toml.load("config.toml")
        self.config = toml.load(self.filename)
        self.cache_model_filename = self.config["cache"]["model_filename"]
        self.cache_main_filename = self.config["cache"]["main_filename"]
        self.cache_pd_filename = self.config["cache"]["pd_filename"]        
