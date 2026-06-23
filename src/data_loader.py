import fastf1
import pandas as pd

fastf1.Cache.enable_cache("data/cache")

SEASONS = [2023, 2024, 2025]


def load_race_laps(year, round_number):
    session = fastf1.get_session(year, round_number, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)

    laps = session.laps.copy()
    laps["Season"] = year
    laps["Round"] = round_number
    laps["EventName"] = session.event["EventName"]

    weather = session.weather_data
    if weather is not None and not weather.empty:
        laps["TrackTemp"] = weather["TrackTemp"].mean()
        laps["AirTemp"] = weather["AirTemp"].mean()
    else:
        laps["TrackTemp"] = None
        laps["AirTemp"] = None

    return laps


def load_season_data(seasons=SEASONS):
    all_laps = []
    for year in seasons:
        schedule = fastf1.get_event_schedule(year)
        race_rounds = schedule[schedule["EventFormat"] != "testing"]["RoundNumber"]
        for round_number in race_rounds:
            if round_number == 0:
                continue
            try:
                laps = load_race_laps(year, round_number)
                all_laps.append(laps)
                print(f"Loaded {year} round {round_number}: {len(laps)} laps")
            except Exception as e:
                print(f"Skipped {year} round {round_number}: {e}")

    return pd.concat(all_laps, ignore_index=True)


if __name__ == "__main__":
    df = load_race_laps(2024, 10)
    print(df.head())
    print(f"\nLoaded {len(df)} laps as a smoke test.")
