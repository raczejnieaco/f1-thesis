"""Etap 1: tabela bazowa (1 wiersz = kierowca w wyścigu) + targety + baseline'y.

Uruchomienie z korzenia repo:  python src/build_base_table.py
Wejście:  data/raw/*.csv (rozpakowany formula_one_1950_2024.zip)
Wyjście:  data/processed/base_table.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
FIRST_YEAR = 2003   # pełne dane kwalifikacji dostępne od 2003 r.

# podział czasowy (do dyskusji z promotorem)
TRAIN = (2003, 2018)
VAL = (2019, 2021)
TEST = (2022, 2024)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{name}.csv", na_values=["\\N"])


def build() -> pd.DataFrame:
    races = read("races")[["raceId", "year", "round", "circuitId", "name", "date"]]
    races = races.rename(columns={"name": "race_name"})
    results = read("results")
    quali = read("qualifying")[["raceId", "driverId", "position"]].rename(
        columns={"position": "quali_position"})
    drivers = read("drivers")[["driverId", "driverRef", "dob"]]
    constr = read("constructors")[["constructorId", "constructorRef"]]
    status = read("status")

    df = (results
          .merge(races, on="raceId")
          .merge(quali, on=["raceId", "driverId"], how="left")
          .merge(drivers, on="driverId")
          .merge(constr, on="constructorId")
          .merge(status, on="statusId"))
    df = df[df.year >= FIRST_YEAR].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dob"] = pd.to_datetime(df["dob"])

    # --- targety ---
    df["podium"] = (df.positionOrder <= 3).astype(int)          # klasyfikacja
    df["finish_pos"] = df.positionOrder                         # regresja (też dla DNF)
    # sklasyfikowany = "Finished" lub "+N Lap(s)"; reszta to DNF/DSQ
    df["finished"] = df.status.str.contains(r"^Finished$|^\+\d+ Laps?$").astype(int)

    # kolumny ZNANE DOPIERO PO WYŚCIGU: zostają tylko do liczenia historii (shift!),
    # nigdy jako cechy bieżącego wyścigu
    post_race = ["points", "laps", "time", "milliseconds", "fastestLap", "rank",
                 "fastestLapTime", "fastestLapSpeed", "statusId", "status",
                 "position", "positionText"]
    df = df.rename(columns={c: f"post_{c}" for c in post_race})

    df = df.sort_values(["date", "raceId", "finish_pos"]).reset_index(drop=True)
    df["split"] = np.select(
        [df.year.between(*TRAIN), df.year.between(*VAL), df.year.between(*TEST)],
        ["train", "val", "test"], default="none")
    return df


def baseline_report(df: pd.DataFrame, split: str) -> None:
    d = df[df.split == split]
    # podium: reguła "grid <= 3" (grid=0 = start z alei, nie liczy się)
    pred = d.grid.between(1, 3).astype(int)
    print(f"[{split}] n={len(d)}, podium rate={d.podium.mean():.3f}")
    print(f"  podium baseline grid<=3: P={precision_score(d.podium, pred):.3f} "
          f"R={recall_score(d.podium, pred):.3f} F1={f1_score(d.podium, pred):.3f}")
    # score ciągły do AUC: im niższy grid, tym wyższy score (pit lane -> ostatni)
    grid_adj = d.grid.where(d.grid > 0, d.groupby("raceId").grid.transform("max") + 1)
    print(f"  podium AUC (score=-grid): {roc_auc_score(d.podium, -grid_adj):.3f}")
    # regresja: pozycja końcowa = grid
    mae = (grid_adj - d.finish_pos).abs().mean()
    rho = d.assign(g=grid_adj).groupby("raceId").apply(
        lambda x: spearmanr(x.g, x.finish_pos)[0], include_groups=False).mean()
    print(f"  pozycja baseline (pred=grid): MAE={mae:.2f}  Spearman/wyścig={rho:.3f}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    base = build()
    base.to_csv(OUT / "base_table.csv", index=False)
    print(f"zapisano {OUT/'base_table.csv'}  shape={base.shape}")
    print(base.split.value_counts().to_dict())
    for s in ["train", "val", "test"]:
        baseline_report(base, s)