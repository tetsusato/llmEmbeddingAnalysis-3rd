import unittest
from dataclasses import dataclass
import polars as pl
import numpy as np
from logging import config
from reduce_vectors_test import ReduceVectors, MultiVectorRepresentation
import plotly.graph_objects as go
import hydra
from hydra import compose, initialize

import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)

class TestReduceVectors(unittest.TestCase):
    num_rows = 100 # number of embedding vectors
    emb_dim = 384 # dimension of the each embedding vector
    with initialize(version_base=None, config_path="../conf", job_name=__file__):
        cfg = compose(config_name="config.yaml",
                      overrides=["+io=qwen2", "cache.enable=False"],
                      return_hydra_config=True)

    def test_init(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        

        rv = ReduceVectors(self.cfg,
                           time_delay=1,
                           stride=1,
                           emb_dim = 4)
        rv.set_reduce_func(getattr(mvr, "reduce_vector_takensembedding"))
        #self.assertEqual(self.cache.__class__.__name__, "Cache")

    def test_sampling(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        
        rv = ReduceVectors(self.cfg,
                           time_delay=1,
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
        
        rv = ReduceVectors(self.cfg,
                           time_delay=1,
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
        
        rv = ReduceVectors(self.cfg,
                           time_delay=1,
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
        
        results, corr, bd1, bd2 = rv.proc_tda(emb,
                                    [0, 1])
        print(results)
        print(corr)
        print(bd1)
        print(bd2)

    def test_proc_pca(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        
        
        rv = ReduceVectors(self.cfg,
                           time_delay=1,
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
        
        results, corr, pcalist = rv.proc_pca(emb,
                                    [0, 1])
        print(results)
        print(corr)
        logger.info(f"pcalist={pcalist}")
        np.testing.assert_array_almost_equal(pcalist,
                                     [[ 1.34332191, -0.07406918],
                                      [ 1.48509638,  0.06699818],
                                      [-1.48509638, -0.06699818],
                                      [-1.34332191,  0.07406918]]
                               )
        

    def test_embedding_to_pd(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        

        rv = ReduceVectors(self.cfg,
                           time_delay=1,
                           stride=1,
                           emb_dim = 4)
        rv.set_reduce_func(getattr(mvr, "reduce_vector_takensembedding"))
        emb = pl.DataFrame({
                "embedding1": [[1.0, 2.1, 3.0, 1.9, 1.0, 2.0, 3.0],
                              [1.1, 2.2, 3.1, 2.1, 1.2, 2.1, 3.1]],
                "embedding2": [[1.0, 2.0, 1.0, 2.0, 3.0],
                               [1.1, 2.1, 1.1, 2.1, 3.1]],
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

        bdlist = rv.embedding_to_pd(reduced_vec,
                                sample_idx=[0],
                                labels=labels,
                                cache_enable = False
                               )
        bd1 = [[1.0125    , 1.11850587],
                  [0.8075    , 1.1553327 ]]
        bd2 = [[0.755    , 0.755    ],
                 [0.905   , 1.000641],
                 [0.75     , 1.0369242]]
        np.testing.assert_array_almost_equal(bd1, bdlist[0])
        np.testing.assert_array_almost_equal(bd2, bdlist[1])
    def test_embedding_to_tsne(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        
        
        rv = ReduceVectors(self.cfg,
                           time_delay=1,
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
    def test_embedding_to_pca(self):
        mvr = MultiVectorRepresentation(embedding_dim=4,
                                        n=2,
                                        dimension=3)        
        
        rv = ReduceVectors(self.cfg,
                           time_delay=1,
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
        pca = rv.embedding_to_pca(vec,
                                    sample_idx=[0, 1, 2, 3],
                                    )
        print("pca", pca)
        
