"""
@File   :   ICASSP27_processing.py
@Date   :   25-8-202613:15
@License: See license file in the root of the repository
@Desc   : 

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
from pathlib import Path
import pandas as pd
from spatialSpectrumProcessing import preprocessing
from SpatialPatCorr import extract_patterns
from EmbedCluster import pattern_clustering

config = {"data_folder": "./data",
          "do_spectrum_processing":False,
          "do_pattern_extraction":False,
          "do_pattern_selection":True,
          "do_goodness": True,
          "spectrum_processing": "melcc",
          "pattern_extraction": "extract",
          "pattern_selection": "tsne_kmeans",
          "goodness": "prediction gain"}

"""
    Raw metadata access
"""
fpath ="./data"
aviaries = pd.read_excel("ICASSP27_birds.xlsx", index_col=0)
aviaries = aviaries["preprocessed_new"].unique()
meta = None

for aviary in aviaries:
    config["aviary"] = aviary
    files = sorted([str(x) for x in Path(fpath).rglob(f"*_{aviary}_meta.xlsx")])
    if len(files)==0:
        print(f"No data in {aviary}")
    else:
        print(f"Processing {aviary}")
        meta = pd.read_excel(files[0], index_col=0)
        for f in files[1:]:
            meta = pd.concat([meta, pd.read_excel(f, index_col=0)])
        meta = meta.reset_index(drop=True)

    """
        Signal representations
    """
    if config["do_spectrum_processing"]==False:
        print(f"No spectrum processing for {aviary}")
    else:
        preprocessing(meta, config)

    """
        Pattern extraction
    """
    if config["do_pattern_extraction"] == False:
        print(f"No pattern extraction for {aviary}")
    else:
        extract_patterns(config)
        pass

    """
        Pattern clustering
    """
    if config["do_pattern_selection"] == False:
        print(f"No pattern clustering for {aviary}")
    else:
        pattern_clustering(config)
        pass

    """
        Goodness computation
    """
    if config["do_goodness"] == False:
        print(f"No goodness for {aviary}")
    else:
        # pattern_clustering(config)
        pass

if __name__ == '__main__':
    print('Hello')
