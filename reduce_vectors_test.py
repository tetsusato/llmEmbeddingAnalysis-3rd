import unittest
from dataclasses import dataclass
import polars as pl
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
from visualize_vectors import euclidean_distance, transform, plot
import sys
import optuna
from dataclasses import dataclass
from typing import Callable
from multi_vector_representation import MultiVectorRepresentation
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import os
import time

config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)
progress = logging.getLogger("progress")

class ReduceVectors():

    def __init__(self,
                 cfg: DictConfig,
                 time_delay,
                 stride,
#                 reduce_func: Callable[[pl.dataframe,
#                                        int, # n
#                                        int, # time_delay
#                                        int, # stride
#                                        int, # dimension
#                                       ],
#                                       np.ndarray
#                                       ],

                 emb_dim = None):
        logger.debug("__init__")
        self.cfg = cfg
        #reduce_func_array = [self.reduce_vector_sampling,
        #                     self.reduce_vector_takensembedding]
        
        self.mode = None
        if emb_dim is None:
            self.emb_dim = 384 # dimension of the each embedding vector
        else:
            self.emb_dim = emb_dim
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

        
    def embedding_to_pd(self,
                        embeddings: pl.dataframe,
                        sample_idx: list,
                        labels: list,
                        gui = False) -> list[np.ndarray]:
        """
        args:
            embeddings: pl.dataframe PD化対象の1次元ベクトル
             元々，オリジナルの埋め込みベクトルを受け取って，次元削減することを
             やっていたが，次元削減後のベクトルを受け取るよう変更
             それに従って，[サンプル数, 埋め込みベクトル(1次元)]という入力が
             [サンプル数, 次元削減後埋め込みベクトルの数, dimension=3]と変更
             入力次元が変わった理由は，「高次元ベクトルを多数の低次元ベクトル
             で表す」という思想のため
        """
        """
        fveclist = self.reduce_func(embeddings,
                                    time_delay=self.time_delay,
                                    stride=self.stride,
                                    )
        print("fvec by gitto=", fveclist.__class__)
        """
        
        birth_death_list = []
        pdlist = []
        #for i in range(fveclist.shape[0]):
        #for i in range(0, sample_num):
        logger.debug(f"embeddings={embeddings}")
        for (fvecs, label) in zip(embeddings, labels):
            #fig = go.Figure()

            #fvecs = fveclist[i]
            #print("fvecs shape([fvecrow, 3])=", fvecs.shape) # [fvecrow, 3]
            x = fvecs[:, 0]
            y = fvecs[:, 1]
            z = fvecs[:, 2]
            if gui:
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', name="dummy"))
                fig.update_traces(marker_size = 1)
                fig.update_layout(title = {"text":f"label={label}",
                                           "x": 0.5,
                                           "y": 0.72})
                fig.show()


            filename = f"pointcloud.pdgm"
            ns = time.time_ns()
            id = ns - int(ns / 1000)*1000
            filename = f"pointcloud-{id}.pdgm"
            pd = hc.PDList.from_alpha_filtration(fvecs, save_to=filename,
                                                  save_boundary_map=True)
            pdlist.append(pd)
            # big bug
            #birth_death = np.array(pd.dth_diagram(1).birth_death_times()).reshape(-1, 2)
            birth_death = np.array(pd.dth_diagram(1).birth_death_times()).T
            #print("pdlist=", pd.dth_diagram(1).birth_death_times())
            #print("birth death=", birth_death)
            birth_death_list.append(birth_death)
        logger.debug(f"birdh-deat={birth_death_list}")
        return birth_death_list

    def embedding_to_tsne(self,
                          embeddings: pl.dataframe,
                          sample_idx: list,
                          perplexity: float) -> np.ndarray:
        """
        args:
            embeddings: pl.dataframe
            sample_idx: list
        return:
            np.ndarray
        """
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=13)
        #npvec = embeddings.to_numpy(structured = False)
        npvec = embeddings.to_numpy()
        print("npvec=", npvec)
        #npvec = np.apply_along_axis(lambda x: x[0], 1, npvec)
        npvec = np.array(list(map(lambda x: x, npvec)))
        print("npvec=", npvec)
        #npvec = npvec[sample_idx]
        fveclist = tsne.fit_transform(npvec)
        #print("fvec by tsne=", fveclist)
        return fveclist
    def embedding_to_pca(self,
                          embeddings: pl.dataframe,
                          sample_idx: list,
                          n_components: int = 2) -> np.ndarray:
        """
        args:
            embeddings: pl.dataframe
            sample_idx: list
        return:
            np.ndarray
        """
        pca = PCA(n_components=n_components,
                   random_state=13)
        #npvec = embeddings.to_numpy(structured = False)
        npvec = embeddings.to_numpy()
        logger.debug(f"npvec={npvec}")
        #npvec = np.apply_along_axis(lambda x: x[0], 1, npvec)
        npvec = np.array(list(map(lambda x: x, npvec)))
        logger.debug(f"npvec={npvec}")
        #npvec = npvec[sample_idx]
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
    def proc_tda(self,
             vecs: pl.DataFrame,
             sample_idx: list) -> (pl.DataFrame,
                        pl.DataFrame,
                        np.ndarray,
                        np.ndarray):
        """
        args:
            vecs: pl.DataFrame [サンプルサイズ, 埋め込みベクトルサイズ]
            sample_idx: list[int] vecsの中からピックアップするindex
        return:
            results: pl.DataFrame
            corr: pl.DataFrame
            bd1: np.ndarray
            bd2: np.ndarray
        """
        # 基本データの準備
        vecs = vecs[sample_idx]
        vec1 = vecs.select("embedding1")
        vec2 = vecs.select("embedding2")

        num_rows = vecs.select(pl.len()).item()
        labels = vecs.select("label").to_numpy().reshape(num_rows)

        # 高次元ベクトルを多数の低次元ベクトルに変換
        reduced_vec1 = self.reduce_func(vec1,
                                        time_delay=self.time_delay,
                                        stride=self.stride,
                                        )
        #logger.info(f"reduced vec1={reduced_vec1}")
        reduced_vec2 = self.reduce_func(vec2,
                                        time_delay=self.time_delay,
                                        stride=self.stride,
                                        )

        # 多数の低次元ベクトルをPDに変換
        bdlist1 = self.embedding_to_pd(reduced_vec1, sample_idx=sample_idx, labels=labels)
        logger.debug(f"bdlist1={bdlist1}")
        bdlist2 = self.embedding_to_pd(reduced_vec2, sample_idx=sample_idx, labels=labels)
        logger.debug(f"bdlist2={bdlist2}")
        sample_num = len(bdlist1)
        
        ##### 描画を分離する2024/07/06 2:16
        ##fig = plt.figure(figsize=(11, 5))
        ##fig.subplots_adjust(hspace=0.6, wspace=0.4)
        ##fig.suptitle(f"birth-death(delay={self.time_delay},stride={self.stride})")

        # PD同士の距離を計算する
        distance = []
        pdw = PersistenceDiagramWrapper(self.cfg)
        for (i, (bd1, bd2, label)) in enumerate(zip(bdlist1, bdlist2, labels)):
            logger.debug(f"bd1={bd1}")
            logger.debug(f"bd2={bd2}")
            pdobj1 = pdw.createPdObject(bd1)
            pdobj2 = pdw.createPdObject(bd2)
            dis = hc.distance.wasserstein(pdobj1, pdobj2)*1000
            #dis = hc.distance.bottleneck(pdobj1, pdobj2)*1000
            distance.append(dis)
            #label = labels[i]
            ##ax = fig.add_subplot(2, sample_num, i+1)
            ##ax.set_title(f"{i}:{dis:.3f}({label:.2f})")
            ##ax.scatter(x=bd1[:, 0], y=bd1[:, 1])
            ##ax = fig.add_subplot(2, sample_num, i+1+sample_num)
            #ax.set_title(f"{i}:{dis:.3f}({label:.2f})")
            #3ax.scatter(x=bd2[:, 0], y=bd2[:, 1])
        ##fig.show()
        column_name = f"dis({self.time_delay}, {self.stride})"
        results = vecs.with_columns(
                              pl.DataFrame(
                                  distance,
                                  schema={column_name: float}))
        #logger.info(f"results: {results}")
        corr = results.select(pl.corr("label", column_name, method="pearson").alias("pearson"),
                              pl.corr("label", column_name, method="spearman").alias("spearman"))
        #print(f"corr = {corr}")
        
        return results, corr, bdlist1, bdlist2
    def proc_draw(self,
                  pd1: list[np.ndarray],
                  pd2: list[np.ndarray],
                  labels: np.ndarray,
                  distance: list[float],
                  sample_idx: list) -> pl.DataFrame:
        pl.Config.set_tbl_cols(-1)
        pl.Config.set_fmt_str_lengths(100)


        num_rows = vecs.select(pl.len()).item()
        #labels = vecs.select("label").to_numpy().reshape(num_rows)
            
        bdlist2 = self.embedding_to_pd(vec2, sample_idx=sample_idx, labels=labels)
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

    def proc_tsne(self, vecs: pl.datatypes.List(pl.Float64), sample_idx, perplexity):
        vecs = vecs[sample_idx]
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
                                          sample_idx=sample_idx,
                                          perplexity=perplexity)
        #print("emb to tsne1=", tsnelist)
        # 元々の2入力に対応した次元削減済みベクトルを得る
        tsnelist1 = tsnelist[0:len(sample_idx)]
        tsnelist2 = tsnelist[len(sample_idx):2*len(sample_idx)]

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
        
        return results, corr, tsnelist
    def proc_pca(self, vecs: pl.datatypes.List(pl.Float64), sample_idx):
        logger.info(f"input={vecs}")
        vecs = vecs[sample_idx]
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
                                          sample_idx=sample_idx,
                                          )
        logger.info(f"pcalist={pcalist}")
        #print("emb to tsne1=", tsnelist)
        # 元々の2入力に対応した次元削減済みベクトルを得る
        pcalist1 = pcalist[0:len(sample_idx)]
        pcalist2 = pcalist[len(sample_idx):2*len(sample_idx)]

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
    def draw_tsne(self, tsne_vecs: np.ndarray, sample_idx):
        vecs = vecs[sample_idx]
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
def objective_tda(cfg: DictConfig):
    def objective(trial: optuna.trial.Trial):


        # ハイパーパラメータの準備
        #time_delay = trial.suggest_int("time_delay", 1, 16)
        time_delay = trial.suggest_int("time_delay",
                                       cfg.hyper_parameter.tda.time_delay_low,
                                       cfg.hyper_parameter.tda.time_delay_high)
        #stride = trial.suggest_int("stride", 1, 16)
        stride = trial.suggest_int("stride",
                                   cfg.hyper_parameter.tda.stride_low,
                                   cfg.hyper_parameter.tda.stride_high)
        number = trial.params
        #reduce_func_index = trial.suggest_int("reduce_func_index", 0, 1)
        reduce_func = trial.suggest_categorical("reduce_func",
                                        ["reduce_vector_sampling",
                                         "reduce_vector_takensembedding"])

        progress.info(f"delay={time_delay}, stride={stride}, reduce_func={reduce_func}")

        # 準備されたハイパーパラメータで一連の処理を実行
        # initの中でハイパーパラメータがインスタンス変数に入る
        rv = ReduceVectors(cfg,
                           time_delay=time_delay,
                           stride=stride,
        #                   reduce_func_index=reduce_func_index
        # この時点では埋め込みベクトルの次元が分からないので，mvrを初期化できない
        #                   reduce_func=getattr(mvr, reduce_func)
                           )
        #pv = PrepareVectors(cfg)
        #vec = pv.getVectors()
        filename = cfg.io.input_filename
        vec = pl.read_parquet(filename)
        embedding_example = vec["embedding1"].to_numpy()
        #logger.info(f"emb1={embedding_example}")
        embedding_dimension = embedding_example.shape[1]
        sample_number = embedding_example.shape[0]
        mvr = MultiVectorRepresentation(embedding_dim=embedding_dimension,
                                            n=embedding_dimension - 3, # 適当
                                            dimension=3)        
        rv.set_reduce_func(getattr(mvr, reduce_func))
        results, corr, pdlist1, pdlist2 = rv.proc_tda(vec, sample_idx)
        #logger.info(f"results in objective_tda={results}")
        #logger.info(f"corr in objective_tda={corr}")
        pearson = corr.select("pearson").head().item()
        spearman = corr.select("spearman").head().item()
        #print(f"results_df={results_df}")

        return pearson, spearman
    return objective
def objective_tsne(cfg: DictConfig):
    def objective(trial: optuna.trial.Trial):
        results_df = pl.DataFrame(
                schema={
                        "perplexity": pl.Int64,
                        "pearson": pl.Float64,
                        "spearman": pl.Float64
                    }
            )
        perplexity = trial.suggest_int("perplexity",
                                       1,
                                       8)
        #reduce_func_index = trial.suggest_int("reduce_func_index", 0, 1)
        reduce_func = trial.suggest_categorical("reduce_func",
                                        ["reduce_vector_takensembedding",
                                         "reduce_vector_takensembedding"])
        #for perplexity in [1, 2, 4, 8]: # should be less than sample=10
        logger.info(f"perplexity={perplexity}")
        mvr = MultiVectorRepresentation()
        #reduce_func_index = 1 # fixed for debug as takens embeddings
        rv = ReduceVectors(time_delay=1,
                           stride=1,
    #                       reduce_func_index=reduce_func_index)
                           reduce_func=getattr(mvr, reduce_func))    
        pv = PrepareVectors("original")
        vec = pv.getVectors(num_rows)
        results, corr, tsne_vecs = rv.proc_tsne(vec, sample_idx, perplexity)
        pearson = corr.select("pearson").head().item()
        spearman = corr.select("spearman").head().item()


        return pearson, spearman    
    return objective

if __name__ == "__main__":
    @dataclass(frozen=True)
    class Mode:
        name: str
        proc: Callable[[optuna.trial.Trial], list[float]]
        hyper_params: list[str]
    """
    mode_list = [
        Mode("tda",
             objective_tda,
             ["values_0",
              "values_1",
              "params_stride",
              "params_time_delay",
              "params_reduce_func"
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
    """
    mode_list = [
        Mode("tda",
             objective_tda,
             ["values_0",
              "values_1",
              "params_stride",
              "params_time_delay",
              "params_reduce_func"
              ]
             ),
        ]
    with initialize(version_base=None, config_path="conf", job_name=__file__):
        #cfg = compose(config_name="qwen2")
        #cfg = compose(config_name="sentence-transformes")
        #cfg = compose(config_name="config.yaml", overrides=["+io=qwen2", "hydra.job.chdir=True", "hydra.run.dir=./outputtest"], return_hydra_config=True)
        cfg = compose(config_name="config.yaml", overrides=["+io=qwen2"], return_hydra_config=True)
    logger.info(f"cfg={OmegaConf.to_yaml(cfg)}")

    mysql_user = os.environ["OPT_USER"]
    mysql_pass = os.environ["OPT_PASS"]
    
    #mode = sys.argv[1]
    #num_rows = 100 # number of embedding vectors
    
    # FIT原稿用
    sample_idx = [6, 3, 1, 4, 0]
    #"""
    #for mode in ["TDA", "TSNE"]:
    for mode in mode_list:
        study_name = mode.name
        storage_name = f"sqlite:///{study_name}.db"
        #storage_name = f"mysql://{mysql_user}:{mysql_pass}@localhost/optuna"
        storage_name = f"mysql://{mysql_user}:{mysql_pass}@127.0.0.1/optuna"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            directions=["minimize", "minimize"],
            load_if_exists=True)
        progress.info(f"study={study}")
        study.optimize(mode.proc(cfg), n_trials=30, n_jobs=-1)
        logger.debug(f"dataframe={study.trials_dataframe()}")
        trial_with_higher_score = study.trials_dataframe().sort_values(
                                            ["values_0", "values_1"]).head(10)
        results = trial_with_higher_score[mode.hyper_params]
        progress.info("結果発表!")
        progress.info(results)
        progress.info("BEST!")
        progress.info(study.best_trials)


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

    # 単純にコサイン類似度のみ
    #sims = rv.pl_cosine_similarity(vec.select("embedding1")[sample_idx],
    #                               vec.select("embedding2")[sample_idx])
    #print("cos sim=", sims)

