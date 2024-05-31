import polars as pl

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import numpy as np
from params import Params
from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)

class VisualizeVectors():
    params = Params("config.toml")
    input_data_filename = params.config["io"]["input_filename"]
    def __init__(self):
        self.df = pl.read_parquet(self.input_data_filename)
        print(self.df)

def euclidean_distance(vecs):
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

def plot(algo, vecs):
    #tsne = TSNE(n_components=2, perplexity=10, random_state=13)
    res1 = algo.fit_transform(vec1.to_series())
    res2 = algo.fit_transform(vec2.to_series())
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    colors = plt.cm.viridis(label / 5.0)
    #sns.scatterplot(x=tsne1[:, 0], y=tsne1[:, 1], hue=label[:, 0])
    #sns.scatterplot(x=tsne2[:, 0], y=tsne2[:, 1], hue=label[:, 0])
    g1 = ax[0].scatter(x=res1[:, 0], y=res1[:, 1], c=colors)
    g2 = ax[1].scatter(x=res2[:, 0], y=res2[:, 1], c=colors)
    plt.colorbar(g1, ax=ax[0])
    num_points = len(vec1)
    ids = np.arange(num_points)
    for i in range(num_points):
        ax[0].text(res1[i, 0], res1[i, 1], str(ids[i]), fontsize=8)
        ax[1].text(res2[i, 0], res2[i, 1], str(ids[i]), fontsize=8)
    plt.show()

if __name__ == "__main__":
    vecs = VisualizeVectors().df
    print("len(vecs)=", len(vecs))
    #numeric_vecs = vecs.select("embedding1", "embedding2")
    numeric_vecs = vecs.select("embedding1", "embedding2", "label")
    numeric_vecs = numeric_vecs.with_columns(
                          euclidean_distance(vecs.select("embedding1",
                                                         "embedding2"),
                          )
        )
    print(f"numeric_vecs={numeric_vecs}")
    print(f"to_nump={numeric_vecs.to_numpy().shape}")
    vec1 = numeric_vecs.select("embedding1")
    vec2 = numeric_vecs.select("embedding2")
    label =  numeric_vecs.select("label")
    tsne = TSNE(n_components=2, perplexity=10, random_state=13)
    plot(tsne, vec1, vec2)    

    mds = MDS(n_components=2, random_state=42)
    plot(mds, vec1, vec2)

    pca = PCA(n_components=2)
    plot(pca, vec1, vec2)

    nmf = NMF(n_components=2)
    plot(nmf, vec1, vec2)


