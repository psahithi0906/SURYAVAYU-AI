# ==========================================================
# GridCast AI - Karnataka Renewable Forecasting Dataset Builder
# Fetches Open-Meteo historical weather data
# Creates training CSV for Solar + Wind + Hybrid plants
# ==========================================================

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# ----------------------------------------------------------
# 15 Karnataka Renewable Plant Locations
# ----------------------------------------------------------
plants = [
    (1,"Pavagada Solar Hub","Solar",14.0990,77.2800,2050),
    (2,"Chitradurga Wind Farm","Wind",14.2306,76.3980,450),
    (3,"Ballari Hybrid Park","Hybrid",15.1394,76.9214,700),
    (4,"Gadag Wind Corridor","Wind",15.4310,75.6350,320),
    (5,"Vijayapura Wind Zone","Wind",16.8302,75.7100,280),
    (6,"Raichur Solar Plains","Solar",16.2120,77.3439,600),
    (7,"Koppal Solar Farm","Solar",15.3480,76.1540,350),
    (8,"Davanagere Wind Belt","Wind",14.4644,75.9218,260),
    (9,"Belagavi Wind Ridge","Wind",15.8497,74.4977,500),
    (10,"Kalaburagi Solar Zone","Solar",17.3297,76.8343,520),
    (11,"Bidar Wind Plateau","Wind",17.9133,77.5301,240),
    (12,"Yadgir Solar Field","Solar",16.7700,77.1376,300),
    (13,"Bagalkot Solar Belt","Solar",16.1867,75.6961,410),
    (14,"Haveri Wind Patch","Wind",14.7937,75.4041,210),
    (15,"Shivamogga Hybrid Zone","Hybrid",13.9299,75.5681,330),
]

# ----------------------------------------------------------
# Date Range (2 years hourly)
# ----------------------------------------------------------
start_date = "2024-01-01"
end_date   = "2025-12-31"

# ----------------------------------------------------------
# Fetch Open-Meteo Data
# ----------------------------------------------------------
def fetch_weather(lat, lon):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,cloud_cover,shortwave_radiation,surface_pressure&timezone=auto"

    r = requests.get(url, timeout=60)
    data = r.json()

    df = pd.DataFrame(data["hourly"])
    return df

# ----------------------------------------------------------
# Power Generation Logic
# ----------------------------------------------------------
def solar_output(rad, cloud, capacity):
    cf = (rad / 1000) * (1 - cloud/120)
    cf = np.clip(cf, 0, 1)
    return capacity * cf

def wind_output(ws, capacity):
    if ws < 3:
        return 0
    elif ws < 12:
        return capacity * ((ws-3)/(12-3))**3
    elif ws <= 25:
        return capacity
    else:
        return 0

# ----------------------------------------------------------
# Build Dataset
# ----------------------------------------------------------
all_data = []

for pid,name,ptype,lat,lon,cap in plants:
    print("Fetching:", name)

    try:
        df = fetch_weather(lat, lon)

        gen = []

        for _,row in df.iterrows():
            rad = row["shortwave_radiation"]
            cloud = row["cloud_cover"]
            ws = row["wind_speed_10m"]

            if ptype == "Solar":
                mw = solar_output(rad, cloud, cap)

            elif ptype == "Wind":
                mw = wind_output(ws, cap)

            else:  # Hybrid
                mw = solar_output(rad, cloud, cap*0.6) + wind_output(ws, cap*0.4)

            noise = np.random.normal(0, mw*0.03)
            mw = max(0, mw + noise)

            gen.append(mw)

        df["generation_mw"] = gen
        df["plant_id"] = pid
        df["plant_name"] = name
        df["plant_type"] = ptype
        df["capacity_mw"] = cap

        all_data.append(df)

        time.sleep(1)

    except Exception as e:
        print("Failed:", name, e)

# ----------------------------------------------------------
# Final CSV
# ----------------------------------------------------------
final_df = pd.concat(all_data, ignore_index=True)

cols = [
    "time",
    "plant_id",
    "plant_name",
    "plant_type",
    "capacity_mw",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "shortwave_radiation",
    "surface_pressure",
    "generation_mw"
]

final_df = final_df[cols]

final_df.rename(columns={"time":"timestamp"}, inplace=True)

final_df.to_csv("karnataka_renewable_training_dataset.csv", index=False)

print("Done.")
print("Rows:", len(final_df))
print("Saved: karnataka_renewable_training_dataset.csv")