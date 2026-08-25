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

config = {"data_folder": "./data",
          "spectrum_processing": "stft",
          "pattern_extraction": "extract",
          "pattern_selection": "t-sne k-means",
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
        continue
    print(f"Processing {aviary}")
    meta = pd.read_excel(files[0], index_col=0)
    for f in files[1:]:
        meta = pd.concat([meta, pd.read_excel(f, index_col=0)])
    meta = meta.reset_index(drop=True)

    """
        Signal representations
    """
    if config["spectrum_processing"] == "none":
        print(f"No spectrum processing for {aviary}")
        continue
    else:
        preprocessing(meta, config)

    """
        Pattern extraction
    """
    if config["pattern_extraction"] == "none":
        print(f"No pattern extraction for {aviary}")
        continue
    else:
        # pattern_extraction(config)
        pass

    """
        Pattern clustering
    """
    if config["pattern_clustering"] == "none":
        print(f"No pattern clustering for {aviary}")
        continue
    else:
        # pattern_clustering(config)
        pass

    """
        Goodness computation
    """
    if config["goodness"] == "none":
        print(f"No goodness for {aviary}")
        continue
    else:
        # pattern_clustering(config)
        pass

if __name__ == '__main__':
    print('Hello')
