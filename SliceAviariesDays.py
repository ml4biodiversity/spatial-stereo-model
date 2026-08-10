"""
@File   :   SliceAviariesDays.py
@Date   :   10-8-202608:30
@License: See license file in the root of the repository
@Desc   : This script is a temporary kludge to map the selected
patterns data to files corresponding to aviaries and dates.

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
import os
import torch
import pandas as pd
from pathlib import Path

def select_fields(item):
    name = item["key"].split("/")[0]
    return {"filename":item["key"], "dataname":name, "datetime":item["meta"]["datetime"]}

def reverse_modulo_weekdays_inplace(data):
    data["tmp"] = data["datetime"].apply(lambda x: x.weekday()+1)
    data["weekday_number"] = data.loc[0, "tmp"]
    base = data.loc[0,"tmp"]
    latest = base
    for c1 in range(1,data.shape[0]):
        if data.loc[c1, "tmp"]!=latest:
            base += 1
        latest = data.loc[c1, "tmp"]
        data.loc[c1, "weekday_number"] = base
    del data["tmp"]

"""
    Main script call for the mapping
"""
if __name__ == '__main__':
    dpath = "selected_patterns/"
    outpath = "extracted_patterns/"
    os.makedirs(outpath, exist_ok=True)

    files = sorted([str(x).replace("\\", "/") for x in Path(dpath).rglob("*.pt")])

    allfiles = pd.DataFrame(columns=['pattern_file', 'filename', 'dataname', 'datetime'])
    for f in files:
        d = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
        keys = list(d.keys())
        df = pd.DataFrame({k: select_fields(d[k]) for k in keys}).T
        df["pattern_file"] = f
        allfiles = pd.concat([allfiles, df], ignore_index=True)

    reorg = pd.DataFrame(columns=['pattern_file', 'filename', 'dataname', 'datetime'])
    for aviary in allfiles["dataname"].unique():
        sel_a = allfiles[allfiles["dataname"] == aviary].copy()
        sel_a = sel_a.sort_values("datetime").reset_index(drop=True)
        reverse_modulo_weekdays_inplace(sel_a)
        sel_a["weekday_number"] -= sel_a["weekday_number"].min()
        reorg = pd.concat([reorg, sel_a], ignore_index=True)

    reorg = reorg.reset_index(drop=True)

    groups = reorg.groupby(["dataname", "weekday_number"])

    for gname, gdata in groups:
        data = {}
        for f in gdata["pattern_file"].unique():
            d = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
            dd = {d[k]["key"]: d[k] for k in d.keys()}  # Fix key (number->file name)
            data = data | {k: dd[k] for k in list(gdata[gdata["pattern_file"] == f]["filename"])}
        torch.save(data, outpath + f"{gname[0]}_day_{int(gname[1])}.pt")
