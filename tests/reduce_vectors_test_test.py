import unittest
from dataclasses import dataclass
import polars as pl
import numpy as np
from logging import config
from reduce_vectors_test import ReduceVectors, MultiVectorRepresentation
import plotly.graph_objects as go

import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)

class TestReduceVectors(unittest.TestCase):
    num_rows = 100 # number of embedding vectors
    emb_dim = 384 # dimension of the each embedding vector

    def test_init(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        

        rv = ReduceVectors(time_delay=1,
                           stride=1,
                           emb_dim = 4)
        rv.set_reduce_func(getattr(mvr, "reduce_vector_takensembedding"))
        #self.assertEqual(self.cache.__class__.__name__, "Cache")

    def test_sampling(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        
        rv = ReduceVectors(time_delay=1,
                           stride=1,
                           emb_dim = 4)
        rv.set_reduce_func(getattr(mvr, "reduce_vector_takensembedding"))
        # 単純なsliding windowのテスト
        emb = pl.DataFrame({
                "embedding": [[1.0, 2.0, 3.0, 4.0],
                              [1.1, 2.1, 3.1, 4.1]]
            }
            )
        new_emb = rv.sampling(emb)
        shouldbe = np.array([
                               [[1.0, 2.0, 3.0],
                                [2.0, 3.0, 4.0]],
                               [[1.1, 2.1, 3.1],
                                [2.1, 3.1, 4.1]]
                            ]
                   )
        np.testing.assert_array_equal(new_emb, shouldbe)
        
        emb = pl.DataFrame({
                "embedding1": [[1.0, 2.0, 3.0, 4.0],
                              [1.1, 2.1, 3.1, 4.1]],
                "embedding2": [[1.0, 2.0, 1.0, 2.0],
                               [1.1, 2.1, 1.1, 2.1]],
                "label": [2.3, 2.4]
            }
            )
        new_emb = rv.sampling(emb)
        self.assertIsNone(new_emb)

    def test_giotto_as_sampling(self):
        #rv = ReduceVectors(emb_dim = 4)
        from gtda.time_series import TakensEmbedding
        te = TakensEmbedding(time_delay=1, dimension=3, stride=1)
        # 単純なsliding windowのテスト
        emb = np.array(
                 [[1.0, 2.0, 3.0, 4.0],
                  [1.1, 2.1, 3.1, 4.1]]
            )
        #new_emb = rv.sampling(emb)
        new_emb = te.fit_transform(emb)
        shouldbe = np.array([
                               [[1.0, 2.0, 3.0],
                                [2.0, 3.0, 4.0]],
                               [[1.1, 2.1, 3.1],
                                [2.1, 3.1, 4.1]]
                            ]
                   )
        np.testing.assert_array_equal(new_emb, shouldbe)

        print(new_emb)
        shouldbe = np.array([
                               [[1.0, 2.0, 3.0],
                                [2.0, 3.0, 4.0]],
                               [[1.1, 2.1, 3.1],
                                [2.1, 3.1, 4.1]]
                            ]
                   )
        np.testing.assert_array_equal(new_emb, shouldbe)
        
        
    def test_giotto(self):
        # giotto-tdaによるdelay embeddingの導入
        from gtda.time_series import SlidingWindow
        windows = SlidingWindow(size=3, stride=1)
        embedding1 = np.array(
                         [1.0, 2.0, 3.0, 4.0, 5.0],
            )
        new_emb = windows.fit_transform(embedding1)
        #print(new_emb)
        shouldbe = np.array(
                               [[1.0, 2.0, 3.0],
                                [2.0, 3.0, 4.0],
                                [3.0, 4.0, 5.0],
                                ],
            )
        np.testing.assert_array_equal(new_emb, shouldbe)
        windows = SlidingWindow(size=3, stride=2)
        new_emb = windows.fit_transform(embedding1)
        shouldbe = np.array(
                               [
                                [1.0, 2.0, 3.0],
                                [3.0, 4.0, 5.0],
                                ],
            )
        #print(new_emb)
        np.testing.assert_array_equal(new_emb, shouldbe)
        windows = SlidingWindow(size=3, stride=3)
        new_emb = windows.fit_transform(embedding1)
        # 前の方が切り捨てられる
        shouldbe = np.array(
                               [
                                [3.0, 4.0, 5.0],
                                ],
            )
        np.testing.assert_array_equal(new_emb, shouldbe)
        #print(new_emb)
        from gtda.time_series import SingleTakensEmbedding
        ste = SingleTakensEmbedding(parameters_type="fixed", time_delay=1, dimension=3, stride=1)
        new_emb = ste.fit_transform(embedding1)
        # completely same as the sliding window
        shouldbe = np.array(
                               [[1.0, 2.0, 3.0],
                                [2.0, 3.0, 4.0],
                                [3.0, 4.0, 5.0],
                                ],
            )
        np.testing.assert_array_equal(new_emb, shouldbe)
        ste = SingleTakensEmbedding(parameters_type="fixed", time_delay=2, dimension=3, stride=1)
        embedding1 = np.array(
                         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            )
        new_emb = ste.fit_transform(embedding1)
        shouldbe = np.array(
                               [[1.0, 3.0, 5.0],
                                [2.0, 4.0, 6.0],
                                [3.0, 5.0, 7.0],
                                ],
            )
        np.testing.assert_array_equal(new_emb, shouldbe)
        ste = SingleTakensEmbedding(parameters_type="fixed", time_delay=2, dimension=3, stride=2)
        new_emb = ste.fit_transform(embedding1)
        shouldbe = np.array(
                               [[1.0, 3.0, 5.0],
                                [3.0, 5.0, 7.0],
                                ],
            )
        np.testing.assert_array_equal(new_emb, shouldbe)
        #print(new_emb)
    def test_cossim(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        
        
        rv = ReduceVectors(time_delay=1,
                           stride=1,
                           emb_dim = 4,
                           )
        rv.set_reduce_func(getattr(mvr, "reduce_vector_takensembedding"))
        emb1 = pl.DataFrame({
                "embedding": [[1.0, 2.0, 3.0, 4.0],
                              [1.1, 2.1, 3.1, 4.1]]
            }
            )
        emb2 = pl.DataFrame({
                "embedding": [[1.0, 2.0, 1.0, 2.0],
                              [1.1, 2.1, 1.1, 2.1]]
            }
            )
        sims = rv.pl_cosine_similarity(emb1, emb2)
        print(sims)
        print(sims.to_numpy())
        shouldbe = np.array(
            [[0.92376043],
             [0.92954233]]

            )
        np.testing.assert_array_almost_equal(sims.to_numpy(), shouldbe)

    def test_proc(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        
        
        rv = ReduceVectors(time_delay=1,
                           stride=1,
                           emb_dim = 4,
                           )
        rv.set_reduce_func(getattr(mvr, "reduce_vector_takensembedding"))
        emb = pl.DataFrame({
                "embedding1": [[1.0, 2.0, 3.0, 4.0],
                              [1.1, 2.1, 3.1, 4.1]],
                "embedding2": [[1.0, 2.0, 1.0, 2.0],
                               [1.1, 2.1, 1.1, 2.1]],
                "label": [2.3, 2.4]
            }
            )
        
        results, corr, bd1, bd2 = rv.proc(emb,
                                    [0, 1])
        print(results)
        print(corr)
        print(bd1)
        print(bd2)

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
    def test_reduce_vector_sampling(self):
        np.random.seed(124)
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        
        rv = ReduceVectors(emb_dim = 4,
                           time_delay=1,
                           stride=1,
                           )
        rv.set_reduce_func(getattr(mvr, "reduce_vector_sampling"))
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
        #reduced_vec = rv.reduce_func(vec,
        #                             n=2,
        #                             )
        reduced_vec = mvr.reduce_vector_sampling(vec)
        print("reduced vec=", reduced_vec)
        np.testing.assert_array_equal(reduced_vec,
                                     [[[4.,  2.,  1. ],
                                       [1.,  2.,  3. ]],

                                      [[4.1, 2.1, 1.1],
                                       [1.1, 2.1, 3.1]]]

                    )
    def test_reduce_vector_takensembedding(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        
        
        rv = ReduceVectors(emb_dim = 4,
                           time_delay=1,
                           stride=1,
                           )
        rv.set_reduce_func(getattr(mvr, "reduce_vector_takensembedding"))
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
        #reduced_vec = rv.reduce_vector_takensembedding(vec)
        reduced_vec = mvr.reduce_vector_takensembedding(vec)
        np.testing.assert_array_equal(reduced_vec,
                    [[[1.,  2.,  3. ],
                      [2.,  3.,  4., ]],
                    [[1.1, 2.1, 3.1],
                     [2.1, 3.1, 4.1]]],

                    )
        
    def test_embedding_to_pd(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        

        rv = ReduceVectors(time_delay=1,
                           stride=1,
                           emb_dim = 4)
        rv.set_reduce_func(getattr(mvr, "reduce_vector_takensembedding"))
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
        # 高次元ベクトルを多数の低次元ベクトルに変換
        reduced_vec = rv.reduce_func(vec,
                                        time_delay=rv.time_delay,
                                        stride=rv.stride,
                                        )

        pd = rv.embedding_to_pd(reduced_vec, sample_idx=[0], labels=labels)
        print("pd", pd)
    def test_embedding_to_tsne(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        
        
        rv = ReduceVectors(time_delay=1,
                           stride=1,
                           emb_dim = 4)
        rv.set_reduce_func(getattr(mvr, "reduce_vector_takensembedding"))
        vecs = pl.DataFrame({
                "embedding1": [[1.0, 2.0, 3.0, 4.0],
                              [1.1, 2.1, 3.1, 4.1]],
                "embedding2": [[1.0, 2.0, 1.0, 2.0],
                               [1.1, 2.1, 1.1, 2.1]],
                "label": [2.3, 2.4]
            }
            )
        vec = pl.concat([
            vecs["embedding1"],
            vecs["embedding2"]],
                        how="vertical").rename("embedding")

        print("vec=", vec)
        tsne = rv.embedding_to_tsne(vec,
                                    sample_idx=[0, 1, 2, 3],
                                    perplexity=1.0)
        print("tsne", tsne)
        
