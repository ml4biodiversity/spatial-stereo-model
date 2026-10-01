"""
@File   :   compute_species_aviaries_detections.py
@Date   :   30-9-202616:47
@License: See license fuile in the root of the repository
@Desc   : 

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
import os
from pathlib import Path

import pandas as pd


def select_local(detected, in_here):
    det = [x.split("_")[0] for x in detected.split("\n")]
    sel = list(set(in_here) & set(det))
    if len(sel)==0:
        sel = det[0]
    else:
        sel = sel[0]
    if sel.find("(")>-1:
        sel = sel[:sel.find("(")-1]
    return sel

fpath = "../zoo_metas/meta"

files = list(Path(fpath).rglob("*.xlsx"))
table = pd.DataFrame()

aviaries = pd.read_excel("Aviaries_03092026.xlsx")
aviaries = aviaries.fillna("none")

SELECT_ALL = False

for file in files:
    print(f"Processing {file}")
    d = pd.read_excel(file)
    aviary = str(file).split(os.sep)[-1][:-10]
    in_here = list(aviaries[(aviaries["preprocessed_new"].apply(
        lambda x: x.find(aviary) > -1))]["Scientific name"].values)

    if SELECT_ALL:
        newlist = ("\n".join(list(d["fusion_model_prediction"].values))).split("\n")
        newlist = [x.split("(")[0].strip() for x in newlist]
    else:
        newlist = [select_local(x, in_here) for x in list(d["fusion_model_prediction"].values)]
    dd = pd.DataFrame({"species": newlist})
    tt = dd.pivot_table(index=["species"], values="species", aggfunc="count")
    tt = tt.rename(columns={"species": "count"})
    tt["species"] = list(tt.index)
    tt["aviary"] = str(file).split(os.sep)[-1]
    tt["duration"] = d["datetime"].max()-d["datetime"].min()
    tt["time_density"] = tt["count"]/tt["duration"].iloc[0].total_seconds()
    table = pd.concat([table,tt],axis=0).reset_index(drop=True)
# table.to_excel("all_detections_species_aviaries.xlsx")



for c1 in range(table.shape[0]):
    aviary = table.loc[c1,'aviary'][:-10]
    sc_name = table.loc[c1,"species"].split("_")[0].strip()
    select = aviaries[(aviaries["preprocessed_new"].apply(lambda x: x.find(aviary)>-1))&
                      aviaries["Scientific name"].apply(lambda x: x.find(sc_name) > -1)]
    if select.shape[0]==1:
        table.loc[c1, "genders"] = str(select["Genders (male.female.unknown)"].values[0])
        table.loc[c1, "total"] = select["Total count"].values[0]

table.to_excel("all_detections_species_aviaries.xlsx")

if __name__ == '__main__':
    print('Hello')
