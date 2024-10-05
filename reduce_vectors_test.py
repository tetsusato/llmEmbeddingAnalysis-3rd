import unittest
from dataclasses import dataclass
import pandas as pd
import polars as pl
import matplotlib
matplotlib.rcParams['backend'] = "Agg"
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from logging import config
import logging
from prepare_vectors import PrepareVectors
import plotly.graph_objects as go
import homcloud.interface as hc
from PersistenceDiagramWrapper import PersistenceDiagramWrapper
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#from visualize_vectors import euclidean_distance, transform, plot
from visualize_vectors import euclidean_distance, transform
import sys
import optuna
import torch
from dataclasses import dataclass
from typing import Callable
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing as multiprocessing


from multi_vector_representation import MultiVectorRepresentation
from cache.cache import Cache                                                     

from experiment_mode import Mode

import os
import time

logger = logging.getLogger(__name__)
progress = logging.getLogger("progress")
results_logger = logging.getLogger("results")


#print(f"logger={logger}")
#print(f"progress={progress}")


class ReduceVectors():

    def __init__(self,
                 cfg: DictConfig,
                 time_delay,
                 stride,
                 emb_dim = None,
                 cache = None
                ):
        logger.debug("__init__")
        self.cfg = cfg
        #reduce_func_array = [self.reduce_vector_sampling,
        #                     self.reduce_vector_takensembedding]
        
        self.emb_dim = cfg.io.embedding_vector_dimension
        self.time_delay = time_delay
        self.stride = stride
        """
        self.reduce_func: Callable[[pl.dataframe,
                                        int, # n
                                        int, # time_delay
                                        int, # stride
                                        int, # dimension
                                        ],
#                                       np.ndarray] = reduce_func_array[reduce_func_index]
                                       np.ndarray] = reduce_func
        """
        self.cache_enable = cfg.cache.enable
        self.cache_reset = cfg.cache.reset
        self.reset = cfg.cache.reset
        """
        self.cache = Cache(cfg,
                           f"ReduceVectors-{",
                           reset = self.reset,
                          )
        """
        if cache is not None:
            self.cache = cache
    def set_reduce_func(self,
                        reduce_func: Callable[[pl.dataframe,
                                               int, # n
                                               int, # time_delay
                                               int, # stride
                                               int, # dimension
                                              ],
                                              np.ndarray
                                              ],
                        ):
        self.reduce_func: Callable[[pl.dataframe,
                                        int, # n
                                        int, # time_delay
                                        int, # stride
                                        int, # dimension
                                        ],
#                                       np.ndarray] = reduce_func_array[reduce_func_index]
                                       np.ndarray] = reduce_func
        
    def sampling(self, embeddings: pl.dataframe) -> pl.dataframe:
        """
        args:
            embeddings: 2d vectors of [:, self.emb_dim]
        """
        print("embedding", embeddings)
        print("embedding shape:", embeddings.shape)
        cols = embeddings.shape[1]
        if cols != 1:
            print("embeddings shape should be [:, 1]")
            return None
        # 細かい成分の処理はnumpyの方がやりやすいので，変換して処理する
        npvec = embeddings.to_numpy(structured = False)
        npvec = np.apply_along_axis(lambda x: x[0], 1, npvec)
        rdim = 3 # reduced dimension
        fvecrow = self.emb_dim - rdim + 1 # SlidingWindowじゃなかったら，変える
        sample_num = npvec.shape[0]
        #sample_num = 7 # for test
        fveclist = np.empty([0, fvecrow, rdim]) # feature vector
        for i in range(sample_num):
            fvec = np.empty([0, rdim])
            row = npvec[i]
            #print("row=", row)
            for j in range(fvecrow):
                rvec = row[j: j + rdim]
                #print("reduced:", rvec)
                fvec = np.vstack([fvec, rvec])
                #print("fvec shape=", fvec.shape)
            fveclist = np.concatenate([fveclist, fvec[None, :, :]], axis=0)
        print("fveclist shape= ", fveclist.shape) # [self.num_rows, fvecrow, rdim]
        return fveclist


    def get_pd(self,
               fvecs_list: list[np.ndarray],
               label_list: list[float],
               dth: int,
               n: int,
               cache_enable,
               gui,
               ) -> np.ndarray:
        birth_death_list = []
        for (fvecs, label) in zip(fvecs_list, label_list):
            x = fvecs[:, 0]
            y = fvecs[:, 1]
            z = fvecs[:, 2]
            """
            if gui:
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', name="dummy"))
                fig.update_traces(marker_size = 1)
                fig.update_layout(title = {"text":f"label={label}",
                                           "x": 0.5,
                                           "y": 0.72})
                fig.show()

            """
            filename = f"pointcloud.pdgm"
            ns = time.time_ns()
            id = ns - int(ns / 1000)*1000
            filename = f"pointcloud-{id}.pdgm"
            import hashlib
            fvecs_hash = hashlib.sha256(fvecs.tobytes()).hexdigest()
            cache_key = f"{fvecs_hash}:{label}"
            tag = "embedding_to_pd"
            if cache_enable:
                val = self.cache.get(cache_key, tag=tag)
            else:
                val = None
            if val is None:
                #pdlist = hc.PDList.from_alpha_filtration(fvecs, save_to=filename,
                pdlist = hc.PDList.from_alpha_filtration(fvecs, save_to=None,
                                                  save_boundary_map=True)
                #logger.debug(f"pdlist={pdlist.shape}")
                #dth = self.cfg.hyper_parameter.tda.dth
                pd = pdlist.dth_diagram(dth)
                #logger.debug(f"pd={pd.shape}")
                birth_death = np.array(pd.birth_death_times()).T
                # y-xの大きさでソート
                sorted_data = birth_death[np.argsort(birth_death[:, 1] - birth_death[:, 0])[::-1]]
                # 大きい方から上位n個の要素を取得
                birth_death = sorted_data[:n]
                #logger.info(f"sorted pd=\n{pd}")
                # xでソート
                birth_death = birth_death[np.argsort(birth_death[:, 0])]
                #logger.debug(f"birth-death={birth_death.shape}")
                #logger.debug(f"set cache (key={cache_key}, tag={tag}, val={birth_death.shape})")
            else:
                birth_death = val
            if cache_enable:
                self.cache.set(cache_key, birth_death, tag=tag)
            birth_death_list.append(birth_death)
        #return birth_death # [ターゲットによるベクトル数, 2]
        return birth_death_list # 2次元のnumpyのリストのはず
        
    def embedding_to_pd(self,
                        embeddings: pl.dataframe,
                        labels: list,
                        n, 
                        gui = False,
                        cache_enable = True
                       ) -> list[np.ndarray]:
        """
        args:
            embeddings: pl.dataframe PD化対象の1次元ベクトル
             元々，オリジナルの埋め込みベクトルを受け取って，次元削減することを
             やっていたが，次元削減後のベクトルを受け取るよう変更
             それに従って，[サンプル数, 埋め込みベクトル(1次元)]という入力が
             [サンプル数, 次元削減後埋め込みベクトルの数, dimension=3]と変更
             入力次元が変わった理由は，「高次元ベクトルを多数の低次元ベクトル
             で表す」という思想のため
             本実装では，embeddingsにembedding1とembedding2が縦にスタックされてくる
             labelsも縦にスタックされてくるように変更
        """
        progress.info("***** embedding_to_pd *****")
        
        birth_death_list = []
        pdlist = []
        #for i in range(fveclist.shape[0]):
        #for i in range(0, sample_num):
        logger.debug(f"input embeddings={embeddings.shape}")
        val = None
        import hashlib
        hashkey = hashlib.sha256(embeddings.tobytes()).hexdigest()
        if cache_enable:
            val = self.cache.get(hashkey)
            """
            if val is not None:
                progress.info(f"Cache is enable. val shape=({val[0].shape}, {val[1].shape})")
            else:
                progress.info(f"Cache is enable. val={val}") # None
            """
        if val is None:
            progress.info(f"Cache is not found. Calculation starts.")
            num_pools = self.cfg.execute.multiprocessing_pools
            #argslist = []
            argslist = [[
                           [], # fvecs
                           [], # label
                           self.cfg.hyper_parameter.tda.dth,
                           n,
                           cache_enable,
                           gui,
                       ]] * num_pools
            logger.debug(f"len zip(embeddings, labels)={len(list(zip(embeddings, labels)))}")
            for (i, (fvecs, label)) in enumerate(zip(embeddings, labels)):
                logger.debug(f"embedding for each sentence pair={fvecs.shape}({fvecs.__class__})")
                argslist[i % num_pools][0].append(fvecs)
                argslist[i % num_pools][1].append(label)
            logger.debug(f"len(argslist)={len(argslist)}")
            multiprocessing.set_start_method("spawn", force=True)
            with multiprocessing.Pool(self.cfg.execute.multiprocessing_pools) as pool:
                """
                for args in argslist:
                    birth_death = self.get_pd(*args)
                    birth_death_list.append(birth_death)
                """
                birth_death_list_list = pool.starmap(self.get_pd, argslist)
            pool.close()
            # 2次元のnumpyのリストのリストのはず
            logger.debug(f"len birth_death_list={len(birth_death_list_list)}")
            birth_death_list = []
            for a_birth_death_list in birth_death_list_list:
                for birth_death in a_birth_death_list:
                    birth_death_list.append(birth_death)
            # birth-deathは，PD毎にに数が異なるので，テンソルにはできない
            #logger.debug(f"np birth_death_list={np.array(birth_death_list)}")
            """
            if cache_enable:
                progress.info(f"Cache is enable. set key={hashkey}, val.shape=({birth_death_list[0].shape}, {birth_death_list[1].shape})")
                self.cache.set(hashkey, birth_death_list)
            """
        else:
            progress.info(f"Cache is accepted. val=({val[0].shape}, {val[1].shape})")
            birth_death_list = val
        logger.debug(f"birth-death shape list:")
        for bd in birth_death_list:
            logger.debug(f"birth-death shape={bd.shape}")
        
        return birth_death_list

    def embedding_to_tsne(self,
                          embeddings: pl.dataframe,
                          perplexity: float) -> np.ndarray:
        """
        args:
            embeddings: pl.dataframe
        return:
            np.ndarray
        """
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=13)
        #npvec = embeddings.to_numpy(structured = False)
        npvec = embeddings.to_numpy()
        #print("npvec=", npvec)
        #npvec = np.apply_along_axis(lambda x: x[0], 1, npvec)
        npvec = np.array(list(map(lambda x: x, npvec)))
        #print("npvec=", npvec)
        fveclist = tsne.fit_transform(npvec)
        #print("fvec by tsne=", fveclist)
        return fveclist
    def embedding_to_pca(self,
                          embeddings: pl.dataframe,
                          n_components: int = 2) -> np.ndarray:
        """
        args:
            embeddings: pl.dataframe
        return:
            np.ndarray
        """
        pca = PCA(n_components=n_components,
                   random_state=13)
        #npvec = embeddings.to_numpy(structured = False)
        npvec = embeddings.to_numpy()
        #logger.debug(f"npvec={npvec}")
        #npvec = np.apply_along_axis(lambda x: x[0], 1, npvec)
        npvec = np.array(list(map(lambda x: x, npvec)))
        #logger.debug(f"npvec={npvec}")
        fveclist = pca.fit_transform(npvec)
        #print("fvec by tsne=", fveclist)
        return fveclist
    
    def pl_cosine_similarity(self, vec1: pl.DataFrame, vec2: pl.DataFrame):
        rows = vec1.select(pl.len()).item()
        print("rows=", rows)
        vec1 = vec1.to_numpy()
        vec1 = np.apply_along_axis(lambda x: x[0], 1, vec1)
        vec2 = vec2.to_numpy()
        vec2 = np.apply_along_axis(lambda x: x[0], 1, vec2)
        print("vec1 to np", vec1)
        sims = []
        for i in range(rows):
            v1 = vec1[i]
            v2 = vec2[i]
            sim = np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1)*np.dot(v2, v2)))
            #print("sim", sim)
            sims.append(sim)
        plsims = pl.DataFrame({
                "cos similarity": sims
            }
            )
        return plsims
    """
    def get_distance(self,
                     pdw: PersistenceDiagramWrapper,
                     bd1: np.ndarray,
                     bd2: np.ndarray,
                     label: float,
                     ) -> float:
        pdobj1 = pdw.createPdObject(bd1)
        pdobj2 = pdw.createPdObject(bd2)
        dis = hc.distance.wasserstein(pdobj1, pdobj2)*1000
        #dis = hc.distance.bottleneck(pdobj1, pdobj2)*1000
        return dis
    """ 
    def get_distance(self,
                     pdw: PersistenceDiagramWrapper,
                     bd1: np.ndarray,
                     bd2: np.ndarray,
                     label: float,
                     ) -> float:
        """
        args:
            bd: [:, 2], 縦に2つのPDの2次元ベクトルがスタックされている想定
        """
        progress.info("***** get_distance *****")
        logger.debug(f"birth-death 1 vector shape={bd1.shape}")
        logger.debug(f"birth-death 2 vector shape={bd2.shape}")
        pdobj1 = pdw.createPdObject(bd1)
        pdobj2 = pdw.createPdObject(bd2)
        dis = hc.distance.wasserstein(pdobj1, pdobj2)*1000
        #dis = hc.distance.bottleneck(pdobj1, pdobj2)*1000

        
        return dis
                     
    def proc_tda(self,
                 vecs: pl.DataFrame,
                 n,
                 cache_enable = True
                ) -> (pl.DataFrame,
                        pl.DataFrame,
                        np.ndarray,
                        np.ndarray):
        """
        args:
            vecs: pl.DataFrame [サンプルサイズ, 埋め込みベクトルサイズ]
        return:
            results: pl.DataFrame
            corr: pl.DataFrame
            bd1: np.ndarray
            bd2: np.ndarray
        """
        progress.info("***** proc_tda *****")
        # 基本データの準備
        #vecs = vecs[sample_idx]
        logger.debug(f"input embedding vectors={vecs.shape}(sentence1, embedding1, sentence2, embedding2, label)")
        vec1 = vecs.select("embedding1")
        vec2 = vecs.select("embedding2")

        num_rows = vecs.select(pl.len()).item()
        labels = vecs.select("label").to_numpy().reshape(num_rows)
        labels = np.concatenate([labels, labels])
        logger.debug(f"labels to input embedding vectors={labels.shape}")
        # 高次元ベクトルを多数の低次元ベクトルに変換
        vec = vec1.rename({"embedding1": "embedding"}).vstack(
              vec2.rename({"embedding2": "embedding"}))
        logger.debug(f"vec: vec1+vec2={vec.shape}")
        reduced_vec = self.reduce_func(vec,
                                        time_delay=self.time_delay,
                                        stride=self.stride,
                                        )
        logger.debug(f"reduced vec={reduced_vec.shape}")
        #logger.debug(f"reduced vec={reduced_vec[0]}")
        # 多数の低次元ベクトルをPDに変換
        bdlist = self.embedding_to_pd(reduced_vec,
                                      labels,
                                      n,
                                      cache_enable
                                     )
        logger.debug(f"birth-death of reduced_vec len={len(bdlist)}")
        ##### 描画を分離する2024/07/06 2:16
        ##fig = plt.figure(figsize=(11, 5))
        ##fig.subplots_adjust(hspace=0.6, wspace=0.4)
        ##fig.suptitle(f"birth-death(delay={self.time_delay},stride={self.stride})")

        #######################
        # PD同士の距離を計算する
        #######################
        #distance = []
        pdw = PersistenceDiagramWrapper(self.cfg)
        argslist = []
        logger.debug(f"stacked labels ={labels}")
        #logger.debug(f"zip={list(zip(bdlist, np.concatenate([labels, labels])))[:3]}")
        # bdlistにはセンテンス1と2が縦にスタックされているので，
        # ラベルもそれぞれに合うようにする
        # 縦にスタックされているのを，横のスタックに直す
        # タプルの方がいいか
        #for (bd, label) in zip(bdlist, labels):
        for i in range(num_rows):
            bd1 = bdlist[i]
            bd2 = bdlist[i + num_rows]
            label = labels[i]
            logger.debug(f"birth-death 1 shape={bd1.shape}")
            logger.debug(f"birth-death 2 shape={bd2.shape}")
            argslist.append([pdw,
                             bd1,
                             bd2,
                             label
                            ]
                           )
        multiprocessing.set_start_method("spawn", force=True)
        #logger.debug(f"argslist={argslist}({len(argslist)})")
        logger.debug(f"len(argslist)={len(argslist)}")
        with multiprocessing.Pool(cfg.execute.multiprocessing_pools) as pool:
          distance = pool.starmap(self.get_distance, argslist)
        #pool.close()
        logger.debug(f"distance list={distance}")
        column_name = f"dis({self.time_delay}, {self.stride})"
        results = vecs.with_columns(
                              pl.DataFrame(
                                  distance,
                                  schema={column_name: float}))
        logger.debug(f"results: {results.sort(column_name)}")
        corr = results.select(pl.corr("label", column_name, method="pearson").alias("pearson"),
                              pl.corr("label", column_name, method="spearman").alias("spearman"))
        logger.info(f"corr = {corr}")
        bdlist1 = bdlist[:len(bdlist)//2]
        bdlist2 = bdlist[len(bdlist)//2:]
        return results, corr, bdlist1, bdlist2
    def proc_draw(self,
                  pd1: list[np.ndarray],
                  pd2: list[np.ndarray],
                  labels: np.ndarray,
                  distance: list[float],
        #          sample_idx: list
                  ) -> pl.DataFrame:
        pl.Config.set_tbl_cols(-1)
        pl.Config.set_fmt_str_lengths(100)


        num_rows = vecs.select(pl.len()).item()
        #labels = vecs.select("label").to_numpy().reshape(num_rows)
            
        #bdlist2 = self.embedding_to_pd(vec2, sample_idx=sample_idx, labels=labels)
        sample_num = len(bdlist1)
        fig = plt.figure(figsize=(11, 5))
        fig.subplots_adjust(hspace=0.6, wspace=0.4)
        fig.suptitle(f"birth-death(delay={self.time_delay},stride={self.stride})")

        pdw = PersistenceDiagramWrapper()
        for (i, (bd1, bd2, label, dis)) in enumerate(zip(bdlist1, bdlist2, labels, distance)):
            ax = fig.add_subplot(2, sample_num, i+1)
            ax.set_title(f"{i}:{dis:.3f}({label:.2f})")
            ax.scatter(x=bd1[:, 0], y=bd1[:, 1])
            ax = fig.add_subplot(2, sample_num, i+1+sample_num)
            x.set_title(f"{i}:{dis:.3f}({label:.2f})")
            ax.scatter(x=bd2[:, 0], y=bd2[:, 1])
        fig.show()

    def proc_tsne(self,
                  vecs: pl.datatypes.List(pl.Float64),
                  perplexity: int,
                  ):
        #vecs = vecs[sample_idx]
        #pl.Config.set_tbl_cols(-1)
        #pl.Config.set_fmt_str_lengths(100)

        # 入力されたベクトルの数．sample_idxの長さに等しい
        num_rows = vecs.select(pl.len()).item()
        labels = vecs.select("label").to_numpy().reshape(num_rows)
            
        # 入力は2カラムあるので，一つにまとめる
        vec = pl.concat([
            vecs["embedding1"],
            vecs["embedding2"]],
                        how="vertical").rename("embedding")

        # TSNEで次元削減した結果のベクトルを得る
        tsnelist = self.embedding_to_tsne(vec,
                                          perplexity=perplexity)
        #print("emb to tsne1=", tsnelist)
        # 元々の2入力に対応した次元削減済みベクトルを得る
        #tsnelist1 = tsnelist[0:len(sample_idx)]
        #tsnelist2 = tsnelist[len(sample_idx):2*len(sample_idx)]
        tsnelist1 = tsnelist[:len(tsnelist)//2]
        tsnelist2 = tsnelist[len(tsnelist)//2:]

        xmin = np.min(tsnelist[:, 0])
        xmax = np.max(tsnelist[:, 0])
        ymin = np.min(tsnelist[:, 1])
        ymax = np.max(tsnelist[:, 1])

        sample_num = len(tsnelist) # これも入力ベクトル数=sample_idxの長さのはず
        #fig = plt.figure(figsize=(18, 5))
        #fig.subplots_adjust(hspace=0.6, wspace=0.4)
        #fig.suptitle(f"tsne: perplexity={perplexity}")
        distance = []
        for (i, (v1, v2)) in enumerate(zip(tsnelist1, tsnelist2)):
            dis = np.linalg.norm(v1 - v2)
            distance.append(dis)
        column_name = "dis"
        distance = pl.DataFrame(
                       distance,
                       schema={
                           column_name: pl.Float64
                           }
                   )
        results = vecs.with_columns(
                    pl.DataFrame(
                        distance,
                        schema={
                           column_name: pl.Float64
                           }
                        )
            )

        logger.debug(f"results: {results.sort("dis")}")
        corr = results.select(pl.corr("label", column_name, method="pearson").alias("pearson"),
                              pl.corr("label", column_name, method="spearman").alias("spearman"))
        #print(f"corr = {corr}")
        
        return results, corr, tsnelist
    def proc_pca(self,
                 vecs: pl.datatypes.List(pl.Float64),
                 ):
        logger.info(f"input={vecs}")
        #vecs = vecs[sample_idx]
        logger.info(f"sampled=={vecs}")
        # 入力されたベクトルの数．sample_idxの長さに等しい
        num_rows = vecs.select(pl.len()).item()
        labels = vecs.select("label").to_numpy().reshape(num_rows)
            
        # 入力は2カラムあるので，一つにまとめる
        vec = pl.concat([
            vecs["embedding1"],
            vecs["embedding2"]],
                        how="vertical").rename("embedding")
        logger.info(f"concatinated=={vec}")
        # PCAで次元削減した結果のベクトルを得る
        pcalist = self.embedding_to_pca(vec,
                                          )
        logger.info(f"pcalist={pcalist}")
        #print("emb to tsne1=", tsnelist)
        # 元々の2入力に対応した次元削減済みベクトルを得る
        #pcalist1 = pcalist[0:len(sample_idx)]
        #pcalist2 = pcalist[len(sample_idx):2*len(sample_idx)]
        pcalist1 = pcalist[:len(pcalist)//2]
        pcalist2 = pcalist[len(pcalist)//2:]

        xmin = np.min(pcalist[:, 0])
        xmax = np.max(pcalist[:, 0])
        ymin = np.min(pcalist[:, 1])
        ymax = np.max(pcalist[:, 1])

        sample_num = len(pcalist) # これも入力ベクトル数=sample_idxの長さのはず
        #fig = plt.figure(figsize=(18, 5))
        #fig.subplots_adjust(hspace=0.6, wspace=0.4)
        #fig.suptitle(f"tsne: perplexity={perplexity}")
        distance = []
        for (i, (v1, v2)) in enumerate(zip(pcalist1, pcalist2)):
            dis = np.linalg.norm(v1 - v2)
            distance.append(dis)
            """
            label = labels[i]
            # 上下に同じデータのembedding1とembedding2に対応した結果をプロット
            ax = fig.add_subplot(2, sample_num, i+1)
            ax.set_title(f"{i}:{dis:.3f}({label:.2f})")
            #ax.set_title(f"{i}:{sim:.3f}({label:.2f})")
            #ax.scatter(x=v1[:, 0], y=v1[:, 1])
            ax.scatter(x=[v1[0]], y=[v1[1]])
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax = fig.add_subplot(2, sample_num, i+1+sample_num)
            #ax.set_title(f"{i}:{dis:.3f}({label:.2f})")
            #ax.set_title(f"{i}:{sim:.3f}({label:.2f})")
            #ax.scatter(x=v2[:, 0], y=v2[:, 1])
            ax.scatter(x=[v2[0]], y=[v2[1]])
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            """
        #fig.show()
        column_name = "dis"
        distance = pl.DataFrame(
                       distance,
                       schema={
                           column_name: pl.Float64
                           }
                   )
        results = vecs.with_columns(
                    pl.DataFrame(
                        distance,
                        schema={
                           column_name: pl.Float64
                           }
                        )
            )

        #logger.info(f"results: {results}")
        corr = results.select(pl.corr("label", column_name, method="pearson").alias("pearson"),
                              pl.corr("label", column_name, method="spearman").alias("spearman"))
        #print(f"corr = {corr}")
        
        return results, corr, pcalist
    #def draw_tsne(self, vecs: pl.datatypes.List(pl.Float64), sample_idx, perplexity):
    def draw_tsne(self,
                  tsne_vecs: np.ndarray,
        #          sample_idx
                  ):
        #vecs = vecs[sample_idx]
        pl.Config.set_tbl_cols(-1)
        pl.Config.set_fmt_str_lengths(100)

        # 入力されたベクトルの数．sample_idxの長さに等しい
        num_rows = vecs.select(pl.len()).item()
        labels = vecs.select("label").to_numpy().reshape(num_rows)
            
        # 元々の2入力に対応した次元削減済みベクトルを得る
        tsnelist1 = tsne_vecs[0:len(sample_idx)]
        tsnelist2 = tsne_vecs[len(sample_idx):2*len(sample_idx)]

        xmin = np.min(tsnelist[:, 0])
        xmax = np.max(tsnelist[:, 0])
        ymin = np.min(tsnelist[:, 1])
        ymax = np.max(tsnelist[:, 1])

        sample_num = len(tsnelist) # これも入力ベクトル数=sample_idxの長さのはず
        fig = plt.figure(figsize=(18, 5))
        fig.subplots_adjust(hspace=0.6, wspace=0.4)
        fig.suptitle(f"tsne: perplexity={perplexity}")
        distance = []
        for (i, (v1, v2)) in enumerate(zip(tsnelist1, tsnelist2)):
            dis = np.linalg.norm(v1 - v2)
            distance.append(dis)
            label = labels[i]
            # 上下に同じデータのembedding1とembedding2に対応した結果をプロット
            ax = fig.add_subplot(2, sample_num, i+1)
            ax.set_title(f"{i}:{dis:.3f}({label:.2f})")
            #ax.set_title(f"{i}:{sim:.3f}({label:.2f})")
            #ax.scatter(x=v1[:, 0], y=v1[:, 1])
            ax.scatter(x=[v1[0]], y=[v1[1]])
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax = fig.add_subplot(2, sample_num, i+1+sample_num)
            #ax.set_title(f"{i}:{dis:.3f}({label:.2f})")
            #ax.set_title(f"{i}:{sim:.3f}({label:.2f})")
            #ax.scatter(x=v2[:, 0], y=v2[:, 1])
            ax.scatter(x=[v2[0]], y=[v2[1]])
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
        fig.show()
def objective_tda(cfg: DictConfig,
                  cache: Cache,
                  vec: pl.DataFrame):
    def objective(trial: optuna.trial.Trial):

        progress.info("***** objective *****")
        # ハイパーパラメータの準備
        #time_delay = trial.suggest_int("time_delay", 1, 16)
        time_delay = trial.suggest_int("time_delay",
                                       cfg.hyper_parameter.tda.time_delay_low,
                                       cfg.hyper_parameter.tda.time_delay_high,
                                       step=cfg.hyper_parameter.tda.time_delay_step)
        #stride = trial.suggest_int("stride", 1, 16)
        stride = trial.suggest_int("stride",
                                   cfg.hyper_parameter.tda.stride_low,
                                   cfg.hyper_parameter.tda.stride_high,
                                   step=cfg.hyper_parameter.tda.stride_step,
                                   )
        n = trial.suggest_int("n",
                                   cfg.hyper_parameter.tda.n_low,
                                   cfg.hyper_parameter.tda.n_high,
                                   step=cfg.hyper_parameter.tda.n_step)
        number = trial.params
        #reduce_func_index = trial.suggest_int("reduce_func_index", 0, 1)
        """
        reduce_func = trial.suggest_categorical("reduce_func",
                                        ["reduce_vector_sampling",
                                         "reduce_vector_takensembedding"])
        """
        reduce_func = trial.suggest_categorical("reduce_func",
                                                cfg.hyper_parameter.tda.embedding_methods
                                                )
        cache_enable = cfg.cache.enable
        progress.info(f"delay={time_delay}, stride={stride}," +
                      f"reduce_func={reduce_func}, n={n}")


        
        # 準備されたハイパーパラメータで一連の処理を実行
        # initの中でハイパーパラメータがインスタンス変数に入る
        rv = ReduceVectors(cfg,
                           time_delay=time_delay,
                           stride=stride,
                           cache=cache
                           )
        embedding_dimension = cfg.io.embedding_vector_dimension
        mvr = MultiVectorRepresentation(embedding_dim=embedding_dimension,
                                            n=embedding_dimension - 3, # 適当
                                            dimension=3)        

        rv.set_reduce_func(getattr(mvr, reduce_func))
        # vec = vec[sample_idx]

        group_num = cfg.execute.group_num
        limit = cfg.execute.limit
        start_index_step = cfg.execute.start_index_step

        pearson_list = []
        spearman_list = []
        # 戦略1
        results, corr, pdlist1, pdlist2 = rv.proc_tda(vec,
                                                      n,
                                                      cache_enable = cache_enable
                                                     )
        #logger.info(f"results in objective_tda={results}")
        #logger.info(f"corr in objective_tda={corr}")
        pearson = corr.select("pearson").head().item()
        spearman = corr.select("spearman").head().item()

        return pearson, spearman
        """
         戦略2
        for i in range(0, limit - group_num, start_index_step):
            subvector = vec[i: i+group_num]
            progress.info(f"execute start = {i}, end = {i+group_num}")
            results, corr, pdlist1, pdlist2 = rv.proc_tda(subvector,
                                                          n,
                                                          cache_enable = cache_enable
                                                         )
            #logger.info(f"results in objective_tda={results}")
            #logger.info(f"corr in objective_tda={corr}")
            pearson = corr.select("pearson").head().item()
            spearman = corr.select("spearman").head().item()
            #print(f"results_df={results_df}")
            pearson_list.append(pearson)
            spearman_list.append(spearman)

        logger.debug(f"pearson before averaging={pearson_list}")
        logger.debug(f"spearman before averaging={spearman_list}")
        average_pearson = sum(pearson_list) / len(pearson_list)
        average_spearman = sum(spearman_list) / len(spearman_list)
        #return pearson, spearman
        return average_pearson, average_spearman
        """
    return objective
def objective_tsne(cfg: DictConfig,
                   cache: Cache,
                   vec: pl.DataFrame):
    def objective(trial: optuna.trial.Trial):
        perplexity = trial.suggest_int("perplexity",
                                       cfg.hyper_parameter.tsne.perplexity_low,
                                       cfg.hyper_parameter.tsne.perplexity_high,
                                      )
        #reduce_func_index = trial.suggest_int("reduce_func_index", 0, 1)
        reduce_func = trial.suggest_categorical("reduce_func",
                                                cfg.hyper_parameter.tsne.embedding_methods
                                                )
        cache_enable = cfg.cache.enable
        progress.info(f"perplexity={perplexity}, reduce_func={reduce_func}")

        rv = ReduceVectors(cfg,
                           time_delay=1,
                           stride=1,
                           )
        # PrepareVectorsは事前に呼ばれていて，ファイル保存されている前提
        embedding_dimension = cfg.io.embedding_vector_dimension
        mvr = MultiVectorRepresentation(embedding_dim=embedding_dimension,
                                            n=embedding_dimension - 3, # 適当
                                            dimension=3)        

        rv.set_reduce_func(getattr(mvr, reduce_func))
        # vec = vec[sample_idx]
        group_num = cfg.execute.group_num
        limit = cfg.execute.limit
        start_index_step = cfg.execute.start_index_step

        pearson_list = []
        spearman_list = []
        #戦略1
        results, corr, tsne_vecs = rv.proc_tsne(vec,
                                                    perplexity)
        pearson = corr.select("pearson").head().item()
        spearman = corr.select("spearman").head().item()
        return pearson, spearman
        
        #戦略2
        """
        #for i in range(0, group_num, num_inputs):
        for i in range(0, limit-group_num+1, start_index_step):
            subvector = vec[i: i+group_num]
            progress.info(f"execute [{i}: {i+group_num}]")
            results, corr, tsne_vecs = rv.proc_tsne(subvector,
                                                    perplexity)
            pearson = corr.select("pearson").head().item()
            spearman = corr.select("spearman").head().item()
            pearson_list.append(pearson)
            spearman_list.append(spearman)
        
        average_pearson = sum(pearson_list) / len(pearson_list)
        average_spearman = sum(spearman_list) / len(spearman_list)
        logger.debug(f"pearson before averaging={pearson_list}")
        logger.debug(f"spearman before averaging={spearman_list}")
        #return pearson, spearman
        return average_pearson, average_spearman
        """
    return objective

mode_list = [
    Mode("tda",
         objective_tda,
         ["values_0",
          "values_1",
          "params_stride",
          "params_time_delay",
          "params_reduce_func",
          "params_n"
          ]
         ),
    Mode("tsne",
         objective_tsne,
         ["values_0",
          "values_1",
          "params_perplexity",
          "params_reduce_func"
          ]
         ),

   ]


if __name__ == "__main__":

    def results_output(
                       corr_index, # values_0 or values 1
                       mode,
                       lang_model,
                       group_num,
                       results,
                       ):
        if mode.name == "tda":
            results_logger.info(f"Results, corr_index, corr, mode, model," +
                                f"group_num, stride, time_delay, reduce_func, n")
            results_corr = results[corr_index].head(1).item()
            results_stride = results['params_stride'].head(1).item()
            results_time_delay = results['params_time_delay'].head(1).item()
            results_reduce_func = results['params_reduce_func'].head(1).item()
            results_n = results['params_n'].head(1).item()
            results_logger.info(f"{corr_index}, {results_corr}, {mode.name}, {lang_model}," +
                                f"{group_num}," +
                                f"{results_stride}, {results_time_delay}, " +
                                f"{results_reduce_func}, {results_n}")
        elif mode.name == "tsne":
            results_logger.info(f"Results, corr_index, corr, mode, model, group_num," +
                                f"perplexity, reduce_func")
            results_corr = results[corr_index].head(1).item()
            results_perplexity = results['params_perplexity'].head(1).item()
            results_reduce_func = results['params_reduce_func'].head(1).item()
            results_logger.info(f"{corr_index}, {results_corr}, {mode.name}, {lang_model}, " +
                                f"{group_num}," +
                                f"{results_perplexity}, " +
                                f"{results_reduce_func}")

        results_logger.info(f"Trial Results({study_name})")
        results_logger.info("結果発表!(values_0優先)")
        results_logger.info(results.sort_values("values_0"))
        results_logger.info("結果発表!(values_1優先)")
        results_logger.info(results.sort_values("values_1"))

        progress.info(f"Trial Results({study_name})")
        progress.info("結果発表!(values_0優先)")
        progress.info(results.sort_values("values_0"))
        progress.info("結果発表!(values_1優先)")
        progress.info(results.sort_values("values_1"))
        progress.info("BEST!")
        progress.info(study.best_trials)
            
            

    results_logger.info(f"START")
    config.fileConfig("logging.conf", disable_existing_loggers = False)


    #lang_model = "qwen2"
    #lang_model = "sentence-transformers"
    with initialize(version_base=None, config_path="conf", job_name=__file__):
        #cfg = compose(config_name="qwen2")
        #cfg = compose(config_name="sentence-transformes")
        #cfg = compose(config_name="config.yaml", overrides=["+io=qwen2", "hydra.job.chdir=True", "hydra.run.dir=./outputtest"], return_hydra_config=True)
        #cfg = compose(config_name="config.yaml", overrides=[f"+io={lang_model}"], return_hydra_config=True)
        cfg = compose(config_name="config.yaml", return_hydra_config=True)
    logger.info(f"cfg={OmegaConf.to_yaml(cfg)}")

    lang_model = cfg.hydra.runtime.choices.io
    label = cfg.label
    
    mysql_user = os.environ["OPT_USER"]
    mysql_pass = os.environ["OPT_PASS"]
    
    #mode = sys.argv[1]
    #num_rows = 100 # number of embedding vectors
    
    # FIT原稿用
    sample_idx = [6, 3, 1, 4, 0]
    #"""
    #for mode in ["TDA", "TSNE"]:
    best_score = pl.DataFrame(
                schema = [
                       ("mode", pl.String),
                       ("values_0", pl.Float64),
                       ("values_1", pl.Float64),
                    ]
                )
    best_results_list = pl.DataFrame(
                schema = [
                       ("mode", pl.String),
                       ("values_0", pl.Float64),
                       ("values_1", pl.Float64),
#                       ("params", pl.String),
                    ]
            )
    for mode in mode_list:
        start_index = cfg.execute.start_index
        start_index_step = cfg.execute.start_index_step
        group_num = cfg.execute.group_num
        limit = cfg.execute.limit

        value0_list = []
        value1_list = []
        # 戦略1
        for index in range(start_index, limit -group_num + 1, start_index_step):
            #study_name = f"{label}-{mode.name}-{lang_model}-{limit}"
            study_name = f"{label}-{mode.name}-{lang_model}-{index}-{limit}"
            storage_name = f"sqlite:///{study_name}.db"
            #storage_name = f"mysql://{mysql_user}:{mysql_pass}@localhost/optuna"
            storage_name = f"mysql://{mysql_user}:{mysql_pass}@127.0.0.1/optuna"
            if cfg.execute.data_reset:
                study_summaries = optuna.study.get_all_study_summaries(storage=storage_name)
                #logger.debug(f"study_summaries={study_summaries}")
                study_exists = any(study.study_name == study_name for study in study_summaries)
                if study_exists:
                    progress.debug(f"Clear study history.")
                    optuna.delete_study(
                        study_name=study_name,
                        storage=storage_name,
                        )
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                directions=["minimize", "minimize"],
                load_if_exists=True)
            progress.info(f"study={study}")
            #study.optimize(mode.proc(cfg), n_trials=1, n_jobs=-1)
            #study.optimize(mode.proc(cfg))
            cache = Cache(cfg,
                          f"ReduceVectors-{study_name}",
                          reset = cfg.cache.reset
                          )
            filename = cfg.io.input_filename
            progress.info(f"input filename = {filename}. Loading...")
            vec = pl.read_parquet(filename)
            progress.info(f"prepared vectors={vec.shape}")
            progress.info(f"group_num={cfg.execute.group_num}")
            group_num = cfg.execute.group_num
            # 戦略1
            vec = vec[index:index + group_num]
            # 戦略2
            #vec = vec[start_index:start_index + limit]
            progress.info(f"target vectors={vec.shape} with start_index={start_index}")
            # 戦略2
            study.optimize(mode.proc(cfg,
                                     cache,
                                     vec),
                           n_trials=cfg.execute.n_trials,
                           n_jobs=10)
            logger.debug(f"dataframe={study.trials_dataframe()}")
            trial_with_higher_score = study.trials_dataframe().sort_values(
                                                ["values_0", "values_1"]).head(10)
            # resultsはpandas dataframe
            results = trial_with_higher_score[mode.hyper_params]

            results_output("values_0",
                           mode,
                           lang_model,
                           group_num,
                           results,
                           )
            results_output("values_1",
                           mode,
                           lang_model,
                           group_num,
                           results,
                           )

            best_results_value0 = results.sort_values("values_0").head(1)
            progress.info(f"best_results_value0={best_results_value0}")
            best_results_value1 = results.sort_values("values_1").head(1)
            progress.info(f"best_results_value1={best_results_value1}")
            value0 = best_results_value0["values_0"].item()
            value1 = best_results_value1["values_1"].item()

            value0_list.append(value0)
            value1_list.append(value1)
            
        value0 = np.sum(value0_list)/len(value0_list)
        value1 = np.sum(value1_list)/len(value1_list)

        # この記録の部分はあとでやる

        params_list = None
        if mode.name == "tda":
            params0 = pd.DataFrame([{
                "mode": mode.name,
                "values_0": value0,
                "values_1": value1,
#                "params": f"params_stride: {best_results_value0['params_stride'].item()}," +
#                      f"params_time_delay: {best_results_value0['params_time_delay'].item()}" 
                }])
            progress.info(f"params0={params0}")
            params1 = pd.DataFrame([{
                "mode": mode.name,
                "values_0": best_results_value1["values_0"].item(),
                "values_1": best_results_value1["values_1"].item(),
#                "params": f"params_stride: {best_results_value1['params_stride'].item()}," +
#                      f"params_time_delay: {best_results_value1['params_time_delay'].item()}" 
                }])

            params_list = pd.concat([params0, params1])
            progress.info(f"params_list = {params_list}")
        elif mode.name == "tsne":
            params0 = pd.DataFrame([{
                "mode": mode.name,
                "values_0": best_results_value0["values_0"].item(),
                "values_1": best_results_value0["values_1"].item(),
#                "params": f"perplexity: {best_results_value0['params_perplexity'].item()}"
                }])

            params1 = pd.DataFrame([{
                "mode": mode.name,
                "values_0": best_results_value1["values_0"].item(),
                "values_1": best_results_value1["values_1"].item(),
#                "params": f"perplexity: {best_results_value1['params_perplexity'].item()}"
                }])

            params_list = pd.concat([params0, params1])

        progress.info(f"params_list={params_list}")

        """
        progress.info(f"params_list.to_list()={params_list.to_list()}")
        progress.info(f"schema={dict(best_results_list.schema)}")
        input_dict =                 {
                                        "mode": mode.name,
                                        "values_0": value0,
                                        "values_1": value1,
                                        "params": [params_list.to_list()]
                                    }
        progress.info(f"input_dict={input_dict}")
        """

        """
        df =  pl.DataFrame(input_dict)
            #                schema=dict(best_results_list.schema)
            schema = [
                   ("mode", pl.String),
                   ("values_0", pl.Float64),
                   ("values_1", pl.Float64),
                   ("params", pl.List(pl.Float64)),
                ]
            )
        """
        #progress.info(f"new df={df}")
        #progress.info(f"new df(polars)={pl.DataFrame(df)}")
        best_results_list = best_results_list.vstack(pl.DataFrame(params_list))

        progress.info("BEST!")
        progress.info(study.best_trials)
        progress.info(f"best value0={best_results_value0}")
        progress.info(f"best value1={best_results_value1}")


    results_logger.info(f"■■■■■最終結果■■■■■■")
    results_logger.info(f"config:\n{cfg}")
    pl.Config.set_fmt_str_lengths(60)
    results_logger.info(f"results={best_results_list}")
    average_resuls = best_results_list.group_by("mode").agg(pl.mean("values_0", "values_1"))

    results_logger.info(f"best average={average_resuls}")
    best_score = best_score.vstack(
                                average_resuls
                               )

    results_logger.info(f"END")
    """
    # 必要なケースを可視化
    for mode in mode_list:
        study_name = mode.name
        storage_name = f"sqlite:///{study_name}.db"
        storage_name = f"mysql://{mysql_user}:{mysql_pass}@127.0.0.1/optuna"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            directions=["minimize", "minimize"],
            load_if_exists=True)
        #study.optimize(mode.proc, n_trials=3)
        logger.debug(f"dataframe={study.trials_dataframe()}")
        trial_with_higher_score = study.trials_dataframe().sort_values(
                                            ["values_0", "values_1"]).head(10)
        results = trial_with_higher_score[mode.hyper_params]
        progress.info("結果発表!")
        progress.info(results)
        progress.info("BEST!")
        progress.info(study.best_trials)
    """
    # 単純にコサイン類似度のみ
    #sims = rv.pl_cosine_similarity(vec.select("embedding1")[sample_idx],
    #                               vec.select("embedding2")[sample_idx])
    #print("cos sim=", sims)

