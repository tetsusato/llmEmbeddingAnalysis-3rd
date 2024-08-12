import polars as pl

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import umap
import numpy as np
import argparse
import textwrap
from params import Params

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf


from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)

class VisualizeVectors():
    """
        VisualizeVectors(mode)
             mode: "original" or "finetuned"
    """
    """
    params = Params("config.toml")
    # 埋め込みベクトルのデータ
    input_data_filename_original = params.config["io"]["input_filename_original"]
    input_data_filename_finetuned = params.config["io"]["input_filename_finetuned"]
    """
    
    def __init__(self, cfg: DictConfig):
        """
            cfgに応じた埋め込みベクトルのデータフレームを返す
        """
        self.input_filename = cfg.io.input_filename
        self.df = pl.read_parquet(self.input_filename)

def euclidean_distance(vecs):
    """
        vecs[0]とvecs[1]の距離をpolarsデータフレームのカラム"distance"で返す
    """
    print("vecs=",vecs)
    vec1 = vecs[:, 0]
    vec2 = vecs[:, 1]
    print("vec1=", vec1)
    print("vec2=", vec2)
    print("np.array(vec1) - np.array(vec2)=", np.array(vec1) - np.array(vec2))
    print("distance=", np.linalg.norm(np.array(vec1) - np.array(vec2), axis=1))
    print("return=",pl.DataFrame(np.linalg.norm(np.array(vec1) - np.array(vec2), axis=1)))
    return pl.DataFrame(np.linalg.norm(np.array(vec1) - np.array(vec2), axis=1),
                        ["distance"])

def transform(algo, vec1: pl.DataFrame, vec2: pl.DataFrame):
    """
       vec1, vec2をalgoのfit_transformに従ってそれぞれ次元削減した
       ベクトルを作る
    """
    vec1 = vec1.to_numpy()
    vec1 = np.apply_along_axis(lambda x: x[0], 1, vec1)
    vec2 = vec2.to_numpy()
    vec2 = np.apply_along_axis(lambda x: x[0], 1, vec2)
    #res1 = pl.DataFrame(algo.fit_transform(vec1.to_series()))
    #res2 = pl.DataFrame(algo.fit_transform(vec2.to_series()))
    res1 = pl.DataFrame(algo.fit_transform(vec1))
    res2 = pl.DataFrame(algo.fit_transform(vec2))
    return (res1, res2)

def plot(algo, vec1, vec2, label):
    #tsne = TSNE(n_components=2, perplexity=10, random_state=13)
    (res1, res2) = transform(algo, vec1, vec2)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    colors = plt.cm.viridis(label / 5.0)
    #sns.scatterplot(x=tsne1[:, 0], y=tsne1[:, 1], hue=label[:, 0])
    #sns.scatterplot(x=tsne2[:, 0], y=tsne2[:, 1], hue=label[:, 0])
    g1 = ax[0].scatter(x=res1[:, 0], y=res1[:, 1], c=colors)
    g2 = ax[1].scatter(x=res2[:, 0], y=res2[:, 1], c=colors)
    ax[0].set_title("\n".join(textwrap.wrap(f"{algo} of Dataset 1", width=30)))
    ax[1].set_title("\n".join(textwrap.wrap(f"{algo} of Dataset 2", width=30)))
    plt.colorbar(g1, ax=ax[0])
    num_points = len(vec1)
    ids = np.arange(num_points)
    for i in range(num_points):
        ax[0].text(res1[i, 0], res1[i, 1], str(ids[i]), fontsize=8)
        ax[1].text(res2[i, 0], res2[i, 1], str(ids[i]), fontsize=8)
    plt.show()

if __name__ == "__main__":
    with initialize(version_base=None, config_path="conf", job_name=__file__):
        #cfg = compose(config_name="qwen2")
        #cfg = compose(config_name="sentence-transformes")
        cfg = compose(config_name="config.yaml", overrides=["+io=qwen2", "hydra.job.chdir=True", "hydra.run.dir=./outputtest"], return_hydra_config=True)
    logger.info(f"cfg={OmegaConf.to_yaml(cfg)}")
    vv = VisualizeVectors(cfg)
    vecs = vv.df
    print("len(vecs)=", len(vecs))
    #numeric_vecs = vecs.select("embedding1", "embedding2")
    numeric_vecs = vecs.select("embedding1", "embedding2", "label")
    numeric_vecs = numeric_vecs.with_columns(
                          euclidean_distance(vecs.select("embedding1",
                                                         "embedding2"),
                          )
        )
    print(f"これは元の空間での距離numeric_vecs={numeric_vecs}")
    print(f"to_nump={numeric_vecs.to_numpy().shape}")
    vec1 = numeric_vecs.select("embedding1")
    vec2 = numeric_vecs.select("embedding2")
    label =  numeric_vecs.select("label")
    tsne = TSNE(n_components=2, perplexity=10, random_state=13)
    tsnevec = transform(tsne, vec1, vec2)

    print(f"tsne空間の距離={tsnevec}")
    plot(tsne, vec1, vec2)
    
    mds = MDS(n_components=2, random_state=42)
    (mdsvec1, mdsvec2) = transform(mds, vec1, vec2)
    mdsvec = pl.DataFrame({
        "vec1": mdsvec1,
        "vec2": mdsvec2
    })

    print("mdsvec=", mdsvec)
    print("distance vec=", euclidean_distance(vecs.select("embedding1",
                                                          "embedding2")))
    mdsvec = mdsvec.with_columns(
                          euclidean_distance(vecs.select("embedding1",
                                                         "embedding2"),
                          )
        )
    print(f"MDS空間の距離={mdsvec}")
    plot(mds, vec1, vec2)

    
    pca = PCA(n_components=2)
    plot(pca, vec1, vec2)

    umap = umap.UMAP(n_components=2, random_state=42)
    plot(umap, vec1, vec2)

    nmf = NMF(n_components=2)
    plot(nmf, vec1, vec2)


