import pandas as pd

COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
VALID_COMPOUNDS = list(COMPOUND_MAP.keys())


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    df = laps.copy()

    df = df.dropna(subset=["LapTime", "Compound", "TyreLife"])
    df = df[df["Compound"].isin(VALID_COMPOUNDS)]
    df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]
    df = df[~df["TrackStatus"].astype(str).str.contains("4|5|6|7", regex=True)]
    df = df[df["IsAccurate"]]

    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()

    median_per_driver = df.groupby("Driver")["LapTimeSeconds"].median()
    df["DriverMedian"] = df["Driver"].map(median_per_driver)
    df = df[df["LapTimeSeconds"] <= df["DriverMedian"] * 1.07]

    df["DegLap"] = df["TyreLife"]
    df["CompoundCode"] = df["Compound"].map(COMPOUND_MAP)

    fastest_per_driver_stint = df.groupby(["Driver", "Stint"])["LapTimeSeconds"].min()
    df = df.join(
        fastest_per_driver_stint.rename("StintFastest"),
        on=["Driver", "Stint"],
    )
    df["LapTimeDelta"] = df["LapTimeSeconds"] - df["StintFastest"]

    keep_cols = [
        "Season", "Round", "EventName", "Driver", "Stint",
        "DegLap", "Compound", "CompoundCode",
        "LapNumber", "TrackTemp", "AirTemp",
        "LapTimeSeconds", "LapTimeDelta",
    ]
    return df[keep_cols].reset_index(drop=True)


if __name__ == "__main__":
    from data_loader import load_race_laps

    raw = load_race_laps(2024, 10)
    clean = clean_laps(raw)
    print(clean.head())
    print(f"\n{len(raw)} raw laps -> {len(clean)} clean laps after filtering.")
