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
from visualize_vectors import euclidean_distance, transform, plot
import sys
import optuna

config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)


class ReduceVectors():

    def __init__(self, time_delay, stride, emb_dim = None):
        logger.debug("__init__")
        self.mode = None
        if emb_dim is None:
            self.emb_dim = 384 # dimension of the each embedding vector
        else:
            self.emb_dim = emb_dim
        self.time_delay = time_delay
        self.stride = stride
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
        print("npvec0", npvec)
        npvec = np.apply_along_axis(lambda x: x[0], 1, npvec)
        print("npvec", npvec)
        print("npvec shape", npvec.shape)
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
                        gui = False):
        """
        npvec1 = embeddings.to_numpy(structured = False)

        npvec1 = np.apply_along_axis(lambda x: x[0], 1, npvec1)
        rdim = 3 # reduced dimension

        fvecrow = self.emb_dim - rdim + 1 # あってる？
        #fvecrow = 5 # for test
        fveclist = np.empty([0, fvecrow, rdim]) # feature vector
        print("empty fveclist", fveclist)
        print("empty fveclist shape", fveclist.shape)
        print("npvec1 shape", npvec1.shape)
        print("npvec1 rows=", npvec1.shape[0])
        sample_num = npvec1.shape[0]
        sample_num = 7
        for i in range(sample_num):
            fvec = np.empty([0, rdim])
            row = npvec1[i]
            #print("row=", row)
            for j in range(fvecrow):
                rvec = row[j: j + rdim]
                #print("reduced:", rvec)
                fvec = np.vstack([fvec, rvec])
                #print("fvec shape=", fvec.shape)
            fveclist = np.concatenate([fveclist, fvec[None, :, :]], axis=0)
        print("fveclist shape= ", fveclist.shape) # [self.num_rows, fvecrow, rdim]
        """
        #sample_num = embeddings.select(pl.len()).item()
        #sample_num = 7 # for test -> sample_idx利用に変更
        #sample_num = 1 # for more test
        #fveclist = self.sampling(embeddings)
        from gtda.time_series import TakensEmbedding
        #te = TakensEmbedding(time_delay=1, dimension=3, stride=1)
        te = TakensEmbedding(time_delay=self.time_delay, dimension=3, stride=self.stride)
        # polars [sample_num, self.emb_dim] to numpy
        npvec = embeddings.to_numpy(structured = False)
        npvec = np.apply_along_axis(lambda x: x[0], 1, npvec)
        #npvec = npvec[sample_idx]

        #labels = labels[sample_idx]
        fveclist = te.fit_transform(npvec)
        #print("fvec by gitto=", fveclist)

        birth_death_list = []
        pdlist = []
        #for i in range(fveclist.shape[0]):
        #for i in range(0, sample_num):
        for (fvecs, label) in zip(fveclist, labels):
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
            pd = hc.PDList.from_alpha_filtration(fvecs, save_to=filename,
                                                  save_boundary_map=True)
            pdlist.append(pd)
            # big bug
            #birth_death = np.array(pd.dth_diagram(1).birth_death_times()).reshape(-1, 2)
            birth_death = np.array(pd.dth_diagram(1).birth_death_times()).T
            #print("pdlist=", pd.dth_diagram(1).birth_death_times())
            #print("birth death=", birth_death)
            birth_death_list.append(birth_death)

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
        #sample_num = embeddings.select(pl.len()).item()
        #sample_num = 7 # for test
        #sample_num = 1 # for more test
        #fveclist = self.sampling(embeddings)
        #te = TakensEmbedding(time_delay=1, dimension=3, stride=1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=13)
        # polars [sample_num, self.emb_dim] to numpy
        #npvec = embeddings.to_numpy(structured = False)
        npvec = embeddings.to_numpy()
        ###npvec = np.apply_along_axis(lambda x: x[0], 1, npvec)
        #npvec = npvec[sample_idx]
        fveclist = tsne.fit_transform(npvec)
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
    def proc(self, vecs: pl.DataFrame, sample_idx: list) -> pl.DataFrame:
        print("**** Normal ****")
        #pv = PrepareVectors("finetuned")
        # ↓これをgetVectorsとcreateRondomSentenceで入れ替える
        #vecs = pv.getVectors(num_rows)
        vecs = vecs[sample_idx]
        #print("vec=", vecs)
        #vec1 = vecs.select("embedding1").to_numpy().reshape(self.num_rows)
        vec1 = vecs.select("embedding1")
        pl.Config.set_tbl_cols(-1)
        pl.Config.set_fmt_str_lengths(100)
        #print("vec1.head(1)", vec1.head(1).to_numpy())

        #vec2 = vecs.select("embedding2").to_numpy().reshape(num_rows)
        vec2 = vecs.select("embedding2")
        num_rows = vecs.select(pl.len()).item()
        labels = vecs.select("label").to_numpy().reshape(num_rows)
            
        # FIT原稿用
        #sample_idx = [6, 3, 1, 4, 0]
        
        bdlist1 = self.embedding_to_pd(vec1, sample_idx=sample_idx, labels=labels)
        bdlist2 = self.embedding_to_pd(vec2, sample_idx=sample_idx, labels=labels)
        #print("original bdlist1=", bdlist1)
        # 全部numpy arrayなわけじゃなかった．．．
        #bdlist1 = np.take(bdlist1, idx, axis=0)
        #bdlist2 = np.take(bdlist2, idx, axis=0)
        # サンプリングはembedding_to_pdの中でやるように変更
        #bdlist1 = [bdlist1[i] for i in sample_idx]
        #bdlist2 = [bdlist2[i] for i in sample_idx]
        #labels = np.take(labels, sample_idx, axis=0)
        sample_num = len(bdlist1)
        ##### 描画を分離する2024/07/06 2:16
        ##fig = plt.figure(figsize=(11, 5))
        ##fig.subplots_adjust(hspace=0.6, wspace=0.4)
        ##fig.suptitle(f"birth-death(delay={self.time_delay},stride={self.stride})")

        distance = []
        pdw = PersistenceDiagramWrapper()
        for (i, (bd1, bd2)) in enumerate(zip(bdlist1, bdlist2)):
            #print("bd=", bd1.shape)
            pdobj1 = pdw.createPdObject(bd1)
            pdobj2 = pdw.createPdObject(bd2)
            dis = hc.distance.wasserstein(pdobj1, pdobj2)*1000
            #dis = hc.distance.bottleneck(pdobj1, pdobj2)*1000
            distance.append(dis)
            label = labels[i]
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
                                  schema={column_name: pl.Float64}))
        #logger.info(f"results: {results}")
        corr = results.select(pl.corr("label", column_name, method="pearson").alias("pearson"),
                              pl.corr("label", column_name, method="spearman").alias("spearman"))
        #print(f"corr = {corr}")
        
        return results, corr
    def proc_tsne(self, vecs: pl.datatypes.List(pl.Float64), sample_idx, perplexity):
        #pv = PrepareVectors("finetuned")
        # ↓これをgetVectorsとcreateRondomSentenceで入れ替える
        #vecs = pv.getVectors(num_rows) 
        #print("vec=", vecs.head(7))
        #vec1 = vecs.select("embedding1").to_numpy().reshape(self.num_rows)
        vecs = vecs[sample_idx]
        pl.Config.set_tbl_cols(-1)
        pl.Config.set_fmt_str_lengths(100)
        #print("vec1.head(1)", vec1.head(1).to_numpy())

        #vec2 = vecs.select("embedding2").to_numpy().reshape(num_rows)
        #
        # 入力されたベクトルの数．sample_idxの長さに等しい
        num_rows = vecs.select(pl.len()).item()
        labels = vecs.select("label").to_numpy().reshape(num_rows)
            
        # limit(5)じゃなくてlimit(7)が正しい
        #offset=7
        """
        vec1 = vecs.select("embedding1")
        vec2 = vecs.select("embedding2")
        vec = vec1.rename({"embedding1": "embedding"}).vstack(
              vec2.rename({"embedding2": "embedding"}))
        """
        # 入力は2カラムあるので，一つにまとめる
        vec = pl.concat([
            vecs["embedding1"],
            vecs["embedding2"]],
                        how="vertical").rename("embedding")
        #print("vstacked vec=", vec)
        # FIT原稿用
        #sample_idx = [6, 3, 1, 4, 0, 6+offset, 3+offset, 1+offset, 4+offset, 0+offset]
        # TSNEで次元削減した結果のベクトルを得る
        tsnelist = self.embedding_to_tsne(vec,
                                          sample_idx=sample_idx,
                                          perplexity=perplexity)
        #print("emb to tsne1=", tsnelist)
        # 元々の2入力に対応した次元削減済みベクトルを得る
        tsnelist1 = tsnelist[0:len(sample_idx)]
        tsnelist2 = tsnelist[len(sample_idx):2*len(sample_idx)]
        # sample_idxの処理はembedding_to_tsneの中でやるように変更
        #tsnelist1 = np.take(tsnelist, sample_idx, axis=0)
        #tsnelist2 = np.take(tsnelist, (np.array(sample_idx)+7).tolist(), axis=0)
        xmin = np.min(tsnelist[:, 0])
        xmax = np.max(tsnelist[:, 0])
        ymin = np.min(tsnelist[:, 1])
        ymax = np.max(tsnelist[:, 1])

        #labels = np.take(labels, sample_idx, axis=0)
        sample_num = len(tsnelist) # これも入力ベクトル数=sample_idxの長さのはず
        fig = plt.figure(figsize=(18, 5))
        fig.subplots_adjust(hspace=0.6, wspace=0.4)
        #fig.suptitle(f"birth-death(delay={self.time_delay},stride={self.stride})")
        fig.suptitle(f"tsne: perplexity={perplexity}")
        distance = []
        for (i, (v1, v2)) in enumerate(zip(tsnelist1, tsnelist2)):
            #print("v1=", v1)
            #dis = hc.distance.wasserstein(pdobj1, pdobj2)*1000
            #dis = hc.distance.bottleneck(pdobj1, pdobj2)*1000
            #sim = np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1)*np.dot(v2, v2)))
            #dis = 1.0 - sim
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

        logger.info(f"results: {results}")
        corr = results.select(pl.corr("label", column_name, method="pearson").alias("pearson"),
                              pl.corr("label", column_name, method="spearman").alias("spearman"))
        #print(f"corr = {corr}")
        
        return results, corr

def objective_tda(trial: optuna.trial.Trial):
    results_df = pl.DataFrame(
            schema={
                    "time delay": pl.Int64,
                    "stride": pl.Int64,                
                    "pearson": pl.Float64,
                    "spearman": pl.Float64
                }
        )
    time_delay = trial.suggest_int("time_delay", 1, 16)
    stride = trial.suggest_int("stride", 1, 16)
    logger.info(f"delay={time_delay}, stride={stride}")
    rv = ReduceVectors(time_delay=time_delay, stride=stride)
    #pv = PrepareVectors("finetuned")
    pv = PrepareVectors("original")
    vec = pv.getVectors(num_rows)
    results, corr = rv.proc(vec, sample_idx)
    pearson = corr.select("pearson").head().item()
    spearman = corr.select("spearman").head().item()
    #print(f"results_df={results_df}")
    """
    results_df = results_df.vstack(pl.DataFrame({"time delay": time_delay,
                               "stride": stride,
                               "pearson": corr.select("pearson").head().item(),
                               "spearman": corr.select("spearman").head().item()
                               }
        ))
    #print(f"results={results_df}")
    print("結果発表!")
    pearson_max_idx = results_df.select(pl.col("pearson").abs().arg_max()).head().item()
    spearman_max_idx = results_df.select(pl.col("spearman").abs().arg_max()).head().item()
    print(results_df[[pearson_max_idx, spearman_max_idx]])
    """
    
    return pearson, spearman

if __name__ == "__main__":
    mode = sys.argv[1]
    num_rows = 100 # number of embedding vectors
    
    # FIT原稿用
    sample_idx = [6, 3, 1, 4, 0]
    #"""
    if mode == "TDA":
        """
        results_df = pl.DataFrame(
                schema={
                        "time delay": pl.Int64,
                        "stride": pl.Int64,                
                        "pearson": pl.Float64,
                        "spearman": pl.Float64
                    }
            )
        time_delay = trial.suggest_int("time_delay", 1, 16)
        stride = trial.suggest_int("stride", 1, 16)
        for time_delay in [1, 2, 4, 8, 16]:
            for stride in [1, 2, 4, 8, 16]:
                logger.info(f"delay={time_delay}, stride={stride}")
                rv = ReduceVectors(time_delay=time_delay, stride=stride)
                #pv = PrepareVectors("finetuned")
                pv = PrepareVectors("original")
                vec = pv.getVectors(num_rows)
                results, corr = rv.proc(vec, sample_idx)
                print(f"results_df={results_df}")
                results_df = results_df.vstack(pl.DataFrame({"time delay": time_delay,
                                           "stride": stride,
                                           "pearson": corr.select("pearson").head().item(),
                                           "spearman": corr.select("spearman").head().item()
                                           }
                    ))
                #print(f"results={results_df}")
        print("結果発表!")
        pearson_max_idx = results_df.select(pl.col("pearson").abs().arg_max()).head().item()
        spearman_max_idx = results_df.select(pl.col("spearman").abs().arg_max()).head().item()
        print(results_df[[pearson_max_idx, spearman_max_idx]])
        """
        study_name = "tda"
        storage_name = f"sqlite:///{study_name}.db"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            directions=["minimize", "minimize"],
            load_if_exists=True)
        study.optimize(objective_tda, n_trials=3)
        logger.debug(f"dataframe={study.trials_dataframe()}")
        trial_with_higher_score = study.trials_dataframe().sort_values(
                                            ["values_0", "values_1"]).head(2)
        results = trial_with_higher_score[["values_0",
                                          "values_1",
                                          "params_stride",
                                           "params_time_delay"]]
        print("結果発表!")
        print(results)
    elif mode == "TSNE":
    #"""
    #"""
        results_df = pl.DataFrame(
                schema={
                        "perplexity": pl.Int64,
                        "pearson": pl.Float64,
                        "spearman": pl.Float64
                    }
            )
        for perplexity in [1, 2, 4, 8]: # should be less than sample=10
            logger.info(f"perplexity={perplexity}")
            rv = ReduceVectors(time_delay=1, stride=1)
            pv = PrepareVectors("original")
            vec = pv.getVectors(num_rows)
            results, corr = rv.proc_tsne(vec, sample_idx, perplexity)
            results_df = results_df.vstack(pl.DataFrame({
                                       "perplexity": perplexity,
                                       "pearson": corr.select("pearson").head().item(),
                                       "spearman": corr.select("spearman").head().item()
                                       }
                ))
        print("結果発表!")
        pearson_max_idx = results_df.select(pl.col("pearson").abs().arg_max()).head().item()
        spearman_max_idx = results_df.select(pl.col("spearman").abs().arg_max()).head().item()
        print(results_df[[pearson_max_idx, spearman_max_idx]])
    #"""
    # 単純にコサイン類似度のみ
    #sims = rv.pl_cosine_similarity(vec.select("embedding1")[sample_idx],
    #                               vec.select("embedding2")[sample_idx])
    #print("cos sim=", sims)

