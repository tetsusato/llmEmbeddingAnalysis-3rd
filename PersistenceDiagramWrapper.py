import time
import os
import importlib


#from cache.cache import Cache
from params import Params


import logging
from logging import config
importlib.reload(config)
config.fileConfig("logging.conf", disable_existing_loggers = False)
logger = logging.getLogger(__name__)

import homcloud.interface as hc
import numpy as np
from scipy.spatial import distance_matrix

#from plot.ipsjplot import plot3d
#from plot.ipsjplot import plot2d


class PersistenceDiagramWrapper():

    def __init__(self):
        params = Params("config.toml")
        self. filename = params.cache_pd_filename

        self.cache_prefix = "PersistenceDiagramWrapper"
        
    def _getPd(self, emb, filename, read_cache_enable=True, write_cache_enable=True,
               force_dim=False,
               dth=None):
        db = Cache(self.filename, prefix=self.cache_prefix)
        pdemb = None
        logger.debug(f"read_cache_enable={read_cache_enable}")
        if read_cache_enable:
            pdemb = db.get(emb, mode="raw_pd")
            logger.debug(f"cache={pdemb}")
        if pdemb is None:
            filename = f"{filename}-{time.time_ns()}"
            start = time.time()
            logger.info(f"PDList from alpha filtration...({time.time() - start})")
            # emb should be a numpy array object.
            pdlist = hc.PDList.from_alpha_filtration(emb, save_to=filename,
                                                     save_boundary_map=True)

            logger.info(f"Getting persistence diagram...({time.time() - start})")
            #logger.info(f"input={emb}")
            #logger.info(f"input[0]={emb[0]}")
            #logger.info(f"input dimension={len(emb[0])}")
            dim = len(emb[0])
            if dth is None:
                if dim == 2:
                    dth = 1
                elif dim == 3:
                    dth = 2
            pd = pdlist.dth_diagram(dth)

            logger.info(f"end...({time.time() - start})")
            logger.debug(f"homcloud pd={pd}")
            birth_death_times = pd.birth_death_times()
            logger.debug(f"Persistence Diagram birth_death_times={birth_death_times}")
            pdemb = np.array(birth_death_times)
            logger.debug(f"Persistence Diagram birth_death_times={pdemb}")
            if write_cache_enable:
                db.set(emb, pdemb, mode="raw_pd")
            os.remove(filename)
        db.close()
        logger.debug(f"Persistence Diagram={pdemb}")
        return pdemb


    def getPd(self, emb, read_cache_enable=True, write_cache_enable=True, force_dim=False,
              dth=1):
        """
        emb: なにかベクトル．このモジュールでは，Sliding Window適用とか考えない
        """
        filename = "pointcloud.pdgm"
        pd = self._getPd(emb, filename,
                         read_cache_enable = read_cache_enable,
                         write_cache_enable = write_cache_enable,
                    force_dim=force_dim,
                    dth=dth)
        return pd.T


    def pdPlot(self, emb):
        plot3d(emb)
        filename = "pointcloud.pdgm"
        pd = self._getPd(emb, filename)
        plot2d(pd)

        return pd.T


    def createPdObject(self, pdlist, degree=1):
        """
        Create homcloud PD object from birth_death_times-numpy array.
        A pdlist will form [[birth1, death1], [birth2, death2], ...]].
        """
        birth = pdlist[:, 0]
        death = pdlist[:, 1]
        pd = hc.PD.from_birth_death(degree, birth, death)
        return pd
