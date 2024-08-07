from logging import config
import numpy as np
import polars as pl
class MultiVectorRepresentation:
    """
        入力されたベクトルを，そのベクトルより低い次元のベクトルの集合に変換する
    """
    def __init__(self,
                 embedding_dim: int = 1,
                 n: int = 1, # samplingで使う．takensでは自動決定なので使わない
                 dimension: int = 3, # この研究では3次元
                 ):
        self.dimension = dimension
        self.n = n
        self.select_indexes = [np.random.choice(embedding_dim,
                                               size=dimension,
                                               replace=False)  for i in range(n)]
        print("select_indexes=", self.select_indexes)
        
    def choice(self,
               src: np.ndarray) -> np.ndarray:
        """
        args:
            src: np.ndarray[self.embedding_dim]
        return:
            [self.n, dimension]
        """
        print("src=", src)
        ret = [[src[i] for i in self.select_indexes[j]] for j in range(self.n)]
        return ret
        
        
    def reduce_vector_sampling(self,
                               embeddings: pl.dataframe,
                               time_delay: int = None,
                               stride: int = None,
                               ) -> np.ndarray:
        """
        args:
            embeddings: pl.dataframe [サンプル数, 埋め込みベクトル(1次元)]
            n: int サンプリングして作るベクトルの数
            time_delay: int Not used in this method
            stride: int Not used in this method
        return:
            np.ndarray: [サンプル数, n, dimension]
        """
        # polars [sample_num, self.emb_dim] to numpy
        npvec = embeddings.to_numpy(structured = False)
        #print(npvec)
        # [入力サンプル数, 埋め込みベクトルの次元]
        npvec = np.apply_along_axis(lambda x: x[0], 1, npvec)
        npveclist = np.empty([npvec.shape[0], self.n, self.dimension])
        # iはnpvecのサンプルを巡る
        #for src, dst in zip(npvec, npveclist) : # [n, dimension]
        for i, src in enumerate(npvec) : # [n, dimension]
            # srcは[embedding_dim]
            """
            for j in range(self.n): # [0, n-1]
                #print("sample dst", dst[j])
                print("sample src", src)
                # dst[j], sampledのサイズはすべて[dimension]
                # sample[i]: [3], size=dimension
                sampled = np.random.choice(src, size=dimension, replace=False)
                #print("=>", sampled)
                dst[j] = sampled
            """
            print("src=", src)
            dst = self.choice(src)
            print("new dst=", dst)
            npveclist[i] = dst

        print("retrun=", npveclist)
        return npveclist
    def reduce_vector_takensembedding(self,
                                      embeddings: pl.dataframe,
                                      time_delay: int = 1,
                                      stride: int = 1,
                                     ) -> np.ndarray:
        """
        args:
            embeddings: pl.dataframe [サンプル数, 埋め込みベクトル(1次元)]
            n: int Not used in this method
            time_delay: int
            stride: int
        return:
            np.ndarray: [サンプル数, TakensEmbeddingの数, dimension=3]
        """
        #logger.info(f"embeddings={embeddings}")
        from gtda.time_series import TakensEmbedding
        #te = TakensEmbedding(time_delay=1, dimension=3, stride=1)
        #te = TakensEmbedding(time_delay=self.time_delay, dimension=3, stride=self.stride)
        te = TakensEmbedding(time_delay=time_delay, dimension=3, stride=stride)
        # polars [sample_num, self.emb_dim] to numpy
        npvec = embeddings.to_numpy(structured = False)
        npvec = np.apply_along_axis(lambda x: x[0], 1, npvec)
        #npvec = npvec[sample_idx]

        #labels = labels[sample_idx]
        fveclist = te.fit_transform(npvec)
        #print("fvec by gitto=", fveclist)
        return fveclist
