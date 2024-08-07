from logging import config
import unittest
from multi_vector_representation import MultiVectorRepresentation
import numpy as np
import polars as pl
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)

class TestMultiVectorRepresetation(unittest.TestCase):
    def test_init(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        

    def test_choice(self):
        np.random.seed(124)
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        
        emb = pl.DataFrame({
                "embedding1": [[1.0, 2.0, 3.0, 4.0],
                              [1.1, 2.1, 3.1, 4.1]],
                "embedding2": [[1.0, 2.0, 1.0, 2.0],
                               [1.1, 2.1, 1.1, 2.1]],
                "label": [2.3, 2.4]
            }
            )
        num_rows = emb.select(pl.len()).item()
        vec = emb.select("embedding1")
        # polars [sample_num, self.emb_dim] to numpy
        npvec = vec.to_numpy(structured = False)
        #print(npvec)
        # [入力サンプル数, 埋め込みベクトルの次元]
        # ここまでやって，embedding1のnparrayができる
        npvec = np.apply_along_axis(lambda x: x[0], 1, npvec)
        ret = mvr.choice(npvec[0])
        print("reduced vec=", ret)
        np.testing.assert_array_equal(ret,
                                      [[4.0, 2.0, 1.0], [1.0, 2.0, 3.0]]
                                      )
        ret = mvr.choice(npvec[1])
        print("reduced vec=", ret)
        np.testing.assert_array_equal(ret,
                                      [[4.1, 2.1, 1.1], [1.1, 2.1, 3.1]]
                                      )
    def test_reduce_vector_takensembedding(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        
        
        emb = pl.DataFrame({
                "embedding1": [[1.0, 2.0, 3.0, 4.0],
                              [1.1, 2.1, 3.1, 4.1]],
                "embedding2": [[1.0, 2.0, 1.0, 2.0],
                               [1.1, 2.1, 1.1, 2.1]],
                "label": [2.3, 2.4]
            }
            )
        num_rows = emb.select(pl.len()).item()
        vec = emb.select("embedding1")
        print("vec=", vec)
        labels = emb["label"].to_numpy().reshape(num_rows)
        reduced_vec = mvr.reduce_vector_takensembedding(vec)
        np.testing.assert_array_equal(reduced_vec,
                    [[[1.,  2.,  3. ],
                      [2.,  3.,  4., ]],
                    [[1.1, 2.1, 3.1],
                     [2.1, 3.1, 4.1]]],

                    )
    def test_reduce_vector_sampling(self):
        np.random.seed(124)
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        
        emb = pl.DataFrame({
                "embedding1": [[1.0, 2.0, 3.0, 4.0],
                              [1.1, 2.1, 3.1, 4.1]],
                "embedding2": [[1.0, 2.0, 1.0, 2.0],
                               [1.1, 2.1, 1.1, 2.1]],
                "label": [2.3, 2.4]
            }
            )
        num_rows = emb.select(pl.len()).item()
        vec = emb.select("embedding1")
        print("vec=", vec)
        labels = emb["label"].to_numpy().reshape(num_rows)
        reduced_vec = mvr.reduce_vector_sampling(vec)
        print("reduced vec=", reduced_vec)
        np.testing.assert_array_equal(reduced_vec,
                                     [[[4.,  2.,  1. ],
                                       [1.,  2.,  3. ]],

                                      [[4.1, 2.1, 1.1],
                                       [1.1, 2.1, 3.1]]]

                    )
        
