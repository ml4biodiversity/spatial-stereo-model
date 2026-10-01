"""
@File   :   compare_goodnesses.py
@Date   :   14-9-202609:48
@License: See license file in the root of the repository
@Desc   : 

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
import pandas as pd
import numpy as np
from pathlib import Path

res = pd.DataFrame()
c0 = 0
for test_name in ["goodnesses","goodnesses_cross"]:
    fpath = f"goodnesses/{test_name}"
    for label in ["stft", "melcc", "logmel", "ccmel"]:
        files = [str(x) for x in Path(fpath).glob(f"*_{label}_*")]
        for f in files:
            d = pd.read_excel(f)
            d["energy_diff"] = d["original_energy_focused"]-d["residual_energy_focused"]
            d["pred_gain"] = 10*np.log10(d["original_energy_focused"]/d["residual_energy_focused"])

            res.loc[c0, "pred_gain_arit"] = d["pred_gain"].mean()
            res.loc[c0, "pred_gain_geom"] = 10*np.log10(d["original_energy_focused"].mean()/d["residual_energy_focused"].mean())
            res.loc[c0, "average_energy_difference"] = d["energy_diff"].mean()
            res.loc[c0, "spectrum"] = label
            res.loc[c0, "filename"] = f
            res.loc[c0, "test"] = test_name
            c0 += 1

import seaborn as sns
import matplotlib.pyplot as plt

sns.pointplot(data=res,x="spectrum",y="pred_gain_arit",hue="test");plt.show()
