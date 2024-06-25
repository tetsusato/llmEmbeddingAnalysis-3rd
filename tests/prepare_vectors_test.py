import unittest
from dataclasses import dataclass
import polars as pl
import numpy as np
from logging import config
from prepare_vectors import PrepareVectors
import plotly.graph_objects as go

import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)

class TestPrepareVectors(unittest.TestCase):
    num_rows = 100 # number of embedding vectors
    emb_dim = 384 # dimension of the each embedding vector

    def test_init(self):
        pv = PrepareVectors("finetuned")
        #self.assertEqual(self.cache.__class__.__name__, "Cache")

    def test_vectors(self):
        pv = PrepareVectors("finetuned")
        vecs = pv.getVectors(self.num_rows)
        print(vecs)
        vec1 = vecs.select("embedding1").to_numpy().reshape(self.num_rows)
        vec2 = vecs.select("embedding2").to_numpy().reshape(self.num_rows)
        labels = vecs.select("label").to_numpy().reshape(self.num_rows)
        sims = []
        for (v1, v2) in zip(vec1, vec2):
            sim = np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1)*np.dot(v2, v2)))
            sims.append(sim)
        print("labels=", labels)
        print("sims=", sims)
        df = pv.describe_each_row(vec1)
        print(df)
        df = pv.describe_each_row(vec2)
        print(df)

        self.assertTrue(np.isclose(labels, [5.,         3.79999995]).all())
        self.assertEqual(sims, [0.9812002934683726, 0.8777169081939417])
        #for (label, sim) in zip(labels, sims):
        #    print(label, sim)

    """
    def test_describe_each_row(self):
        vec = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.describe_each_row(vec)
    """
          
    def test_randomVectors(self):
        pv = PrepareVectors("finetuned")
        df = pv.createRandomSentence(self.num_rows)
        print("random polars dataframe=", df)
        vec1 = df.select("embedding1").to_numpy().reshape(self.num_rows)
        vec2 = df.select("embedding2").to_numpy().reshape(self.num_rows)
        labels = df.select("label").to_numpy().reshape(self.num_rows)
        sims = []
        for (v1, v2) in zip(vec1, vec2):
            sim = np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1)*np.dot(v2, v2)))
            sims.append(sim)
        print("labels=", labels)
        print("sims=", sims)
        df = pv.describe_each_row(vec1)
        print(df)
        df = pv.describe_each_row(vec2)
        print(df)

    def compare_vectors(self):
        print("Compare describes of random and non-random embedding vectors")
        print("**** Random ****")
        pv = PrepareVectors("finetuned")
        df = pv.createRandomSentence(self.num_rows)
        vec1 = df.select("embedding1").to_numpy().reshape(self.num_rows)
        vec2 = df.select("embedding2").to_numpy().reshape(self.num_rows)
        labels = df.select("label").to_numpy().reshape(self.num_rows)
        sims = []
        for (v1, v2) in zip(vec1, vec2):
            sim = np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1)*np.dot(v2, v2)))
            sims.append(sim)
        df = pv.describe_each_row(np.concatenate([vec1, vec2]))
        #print(df)
        print("Mean:")
        print(df["mean"].describe())
        print("Std:")
        print(df["std"].describe())
        print("Min:")
        print(df["min"].describe())
        print("Max:")
        print(df["max"].describe())

        print("**** Normal ****")
        pv = PrepareVectors("finetuned")
        vecs = pv.getVectors(self.num_rows)
        vec1 = vecs.select("embedding1").to_numpy().reshape(self.num_rows)
        vec2 = vecs.select("embedding2").to_numpy().reshape(self.num_rows)
        labels = vecs.select("label").to_numpy().reshape(self.num_rows)
        sims = []
        for (v1, v2) in zip(vec1, vec2):
            sim = np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1)*np.dot(v2, v2)))
            sims.append(sim)
        df = pv.describe_each_row(np.concatenate([vec1, vec2]))
        #print(df)
        print("Mean:")
        print(df["mean"].describe())
        print("Std:")
        print(df["std"].describe())
        print("Min:")
        print(df["min"].describe())
        print("Max:")
        print(df["max"].describe())

    def reduce_vectors(self):
        print("**** Normal ****")
        pv = PrepareVectors("finetuned")
        vecs = pv.getVectors(self.num_rows)
        print("vec=", vecs.head(1))
        #vec1 = vecs.select("embedding1").to_numpy().reshape(self.num_rows)
        vec1 = vecs.select("embedding1")
        pl.Config.set_tbl_cols(-1)
        pl.Config.set_fmt_str_lengths(100)
        #print("vec1.head(1)", vec1.head(1).to_numpy())

        vec2 = vecs.select("embedding2").to_numpy().reshape(self.num_rows)
        labels = vecs.select("label").to_numpy().reshape(self.num_rows)
        sims = []
        #for (v1, v2) in zip(vec1, vec2):
        #    sim = np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1)*np.dot(v2, v2)))
        #    sims.append(sim)

        npvec1 = vec1.to_numpy(structured = False)

        npvec1 = np.apply_along_axis(lambda x: x[0], 1, npvec1)
        rdim = 3 # reduced dimension

        fvecrow = self.emb_dim - rdim + 1 # あってる？
        fveclist = np.empty([0, fvecrow, rdim]) # feature vector
        print("empty fveclist", fveclist)
        print("empty fveclist shape", fveclist.shape)
        print("npvec1 shape", npvec1.shape)
        print("npvec1 rows=", npvec1.shape[0])
        for i in range(npvec1.shape[0]):
            fvec = np.empty([0, rdim])
            row = npvec1[i]
            #print("row=", row)
            fvecrow = len(row) - rdim + 1 # row = self.emb_vecか
            for j in range(fvecrow):
                rvec = row[j: j + rdim]
                #print("reduced:", rvec)
                fvec = np.vstack([fvec, rvec])
            #print("fvec shape=", fvec.shape)
            fveclist = np.concatenate([fveclist, fvec[None, :, :]], axis=0)
        print("fveclist shape= ", fveclist.shape) # [self.num_rows, fvecrow, rdim]

        fig = go.Figure()

        for i in range(fveclist.shape[0]):
            fvec = fveclist[i]
            x = fvec[0]
            y = fvec[1]
            z = fvec[2]
            fig.add_trace(go.Scatter3d(x=x, y=y, z=y, mode='markers', name="dummy"))


        fig.show()
