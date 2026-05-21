###########################################
##   AIS DATA CLEANING AND PROCESSING    ##
###########################################

# --- IMPORT LIBARRIES ---

import pandas as pd
import geopandas as gpd
import duckdb
import requests
import folium
from pathlib import Path
from shapely import wkt
import numpy as np
from haversine import haversine_vector, Unit

# --- CONFIG ---
year = 2020
month = "05"
input_dir = Path(f"D:/AIS_Data/AIS_Monthly_{year}")
output_dir = Path("D:/AIS_Cleaned_Test")

output_dir.mkdir(parents=True, exist_ok=True)

### ----- 1. FLAG POINTS ON LAND ----- ###

# Land layer
# BOEM Renewable States REST URL
rest_url = "https://services7.arcgis.com/G5Ma95RzqJRPKsWL/arcgis/rest/services/BOEM_Renewable_States/FeatureServer/0/query"

# Study Area Bounding Box (to clip the land locally)
# Use the same bounding box that I used for AIS data download
BBOX = {"xmin": -74.05, "ymin": 39.50, "xmax": -68.78, "ymax": 41.69}

def flag_land_points():
    # 1. FETCH LAND DATA
    print("Requesting land data...")
    payload = {
        "where": "1=1", 
        "outFields": "*", 
        "f": "geojson", 
        "outSR": "4326",
        "returnGeometry": "true"
    }
    # Post request
    response = requests.post(rest_url, data=payload)
    data = response.json()

    # Check if the server sent back features
    if "features" not in data:
        print("Error: Server did not return features. Response was:")
        print(data) 
        return

    land_gdf = gpd.GeoDataFrame.from_features(data, crs="EPSG:4326")
    print(f"Fetched {len(land_gdf)} land features.")

    # 2. CLIP & BUFFER LAND DATA
    print("Clipping and buffering land mask...")
    land_gdf = land_gdf.cx[BBOX['xmin']:BBOX['xmax'], BBOX['ymin']:BBOX['ymax']]
    # Project to Conus Albers
    land_gdf = land_gdf.to_crs(epsg=5070)
    # Create -100 meter buffer so
    # Points in harbors will not be flagged
    land_gdf['geometry'] = land_gdf.buffer(-100)
    land_gdf = land_gdf[~land_gdf.is_empty].to_crs(epsg=4326)
    
    # Convert geometry to WKB (well known binary) for DuckDB
    land_gdf['geom_wkb'] = land_gdf['geometry'].apply(lambda x: x.wkb)
    land_df_for_duck = pd.DataFrame(land_gdf.drop(columns='geometry'))

    # 3. DUCKDB SPATIAL JOIN
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.register("buffered_land", land_df_for_duck)

    parquet_files = list(input_dir.glob(f"AIS_{year}_{month}*.parquet"))

    for f_path in parquet_files:
        output_path = output_dir / f"{f_path.stem}_LANDFLAG.csv"
        print(f"Processing {f_path.name} -> {output_path.name}...")

        # DuckDB SQL query to flag rows on land and add notes
        query = f"""
            COPY (
                SELECT 
                    main.* EXCLUDE (geometry_wkb),
                    ST_AsText(ST_GeomFromWKB(main.geometry_wkb)) AS geometry,
                    (l.geom_wkb IS NOT NULL) AS flag_on_land,
                    CASE WHEN l.geom_wkb IS NOT NULL THEN 'Point on land' ELSE NULL END AS cleaning_notes
                FROM read_parquet('{str(f_path)}') AS main
                LEFT JOIN buffered_land AS l
                  ON main.longitude BETWEEN {BBOX['xmin']} AND {BBOX['xmax']}
                  AND main.latitude BETWEEN {BBOX['ymin']} AND {BBOX['ymax']}
                  AND ST_Intersects(
                      ST_GeomFromWKB(main.geometry_wkb), 
                      ST_GeomFromWKB(l.geom_wkb)
                  )
            ) TO '{str(output_path)}' (HEADER, DELIMITER ',');
        """
        
        try:
            con.execute(query)
            
            # --- STATS & MAPPING ---
            # Load only the necessary columns to save memory
            df_check = pd.read_csv(output_path, usecols=['geometry', 'flag_on_land'])
            
            flagged_count = df_check['flag_on_land'].sum()
            total_rows = len(df_check)
            
            print(f"  Total Rows: {total_rows:,}")
            print(f"  Flagged (On Land): {flagged_count:,} ({(flagged_count/total_rows)*100:.2f}%)")

            if flagged_count > 0:
                print("  Generating verification map...")
                # Sample up to 1000 flagged points for the map
                flagged_sample = df_check[df_check['flag_on_land'] == True].sample(min(flagged_count, 1000))
                flagged_sample['geometry'] = flagged_sample['geometry'].apply(wkt.loads)
                gdf_sample = gpd.GeoDataFrame(flagged_sample, crs="EPSG:4326")

                # Create Map
                m = folium.Map(
                    location=[gdf_sample.geometry.y.mean(), gdf_sample.geometry.x.mean()], 
                    zoom_start=11,
                    tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                    attr='Google Satellite'
                )

                for _, row in gdf_sample.iterrows():
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=2, color='red', fill=True
                    ).add_to(m)

                map_path = output_dir / f"{f_path.stem}_Flagged_Map.html"
                m.save(str(map_path))
                print(f"  Map saved: {map_path.name}")
            else:
                print("  No points were flagged; skipping map generation.")

        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == "__main__":
    flag_land_points()

### ----- 2. FLAG POINTS WHERE CALCULATED SOG DIFFERS FROM REPORTED SOG ----- ###

# Stats function
def print_vessel_stats(df):
    print("\n" + "="*80)
    print("VESSEL TYPE PERFORMANCE SUMMARY (Smoothed vs. Reported)")
    print("="*80)

    # Group by vessel type and calculate stats for both speed columns
    # 'sog' is the reported value, 'calc_sog' is our 2-point moving average
    stats = df.groupby('vessel_group').agg({
        'sog': ['median', 'mean', 'std'],
        'moving_avg_sog': ['median', 'mean', 'std']
    })

    # Round for readability (2 decimal places)
    stats = stats.round(2)

    # Flatten the multi-index columns for a cleaner print
    stats.columns = [
        'Rep_SOG_Median', 'Rep_SOG_Mean', 'Rep_SOG_Std',
        'Calc_SOG_Median', 'Calc_SOG_Mean', 'Calc_SOG_Std'
    ]

    print(stats.to_string())
    print("="*80 + "\n")



def calculate_sog(
    input_csv,
    output_csv,
    timestamp_col='base_date_time',
    mmsi_col='mmsi',
    lat_col='latitude',
    lon_col='longitude',
    reported_sog_col='sog'
):
    # 1. LOAD DATA
    df = pd.read_csv(input_csv)
    total_original_points = len(df)

    # 2. PREPARE TIME & SORT
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values([mmsi_col, timestamp_col])

    # 3. CALCULATE SHIFTS (Vectorized)
    # Instead of row-by-row apply, we shift the entire column
    df['prev_lat'] = df.groupby(mmsi_col)[lat_col].shift(1)
    df['prev_lon'] = df.groupby(mmsi_col)[lon_col].shift(1)
    df['prev_time'] = df.groupby(mmsi_col)[timestamp_col].shift(1)

    # Calculate time difference in hours
    df['time_hours'] = (df[timestamp_col] - df['prev_time']).dt.total_seconds() / 3600

    # FILTER: Remove jitter < 2 seconds (0.00055 hours) and the first point of each trip (NaN)
    df = df[df['time_hours'] > (2/3600)].copy()

    # 4. CALCULATE DISTANCE (Vectorized)
    # haversine_vector is ~50x faster than .apply(haversine)
    points_now = df[[lat_col, lon_col]].to_numpy()
    points_prev = df[['prev_lat', 'prev_lon']].to_numpy()
    
    df['distance_nm'] = haversine_vector(points_now, points_prev, unit=Unit.NAUTICAL_MILES)

    # 5. CALCULATE RAW & AVERAGED SOG
    # Initial point-to-point calculation
    df['raw_calc_sog'] = df['distance_nm'] / df['time_hours']

    # 2-point Moving Average (The "Smoother")
    # This averages the current segment speed and the previous one per vessel
    df['moving_avg_sog'] = df.groupby(mmsi_col)['raw_calc_sog'].transform(
        lambda x: x.rolling(window=2, min_periods=1).mean()
    )

    # 6. FLAG ANOMALIES (The 0.5 Knot Gate)
    # Define the gate: Only care if reported speed > 0.5
    valid_mask = df[reported_sog_col] > 0.5

    # Initialize columns
    df['percent_difference'] = np.nan
    df['flag_10_percent'] = False
    df['flag_30_percent'] = False

    # Calculate difference using the moving average
    # We use (reported + 0.001) to safely handle any weird zeros
    df.loc[valid_mask, 'percent_difference'] = (
        (abs(df.loc[valid_mask, 'moving_avg_sog'] - df.loc[valid_mask, reported_sog_col])) / 
        (df.loc[valid_mask, reported_sog_col] + 0.001)
    ) * 100

    # Set Flags
    df.loc[valid_mask, 'flag_10_percent'] = df.loc[valid_mask, 'percent_difference'] > 10
    df.loc[valid_mask, 'flag_30_percent'] = df.loc[valid_mask, 'percent_difference'] > 30

    # 7. CLEANUP & EXPORT
    # Drop helper columns to keep CSV clean
    cols_to_drop = ['prev_lat', 'prev_lon', 'prev_time', 'time_hours', 'distance_nm']
    df_final = df.drop(columns=cols_to_drop)
    
    df_final.to_csv(output_csv, index=False)

    # SUMMARY PRINT
    print("\n" + "="*50)
    print(" --------------- AIS SOG ANALYSIS ----------------------")
    print(f" Total input points:     {total_original_points:,}")
    print(f" Processed points:       {len(df_final):,}")
    print(f" Ignored (SOG <= 0.5):   {(df[reported_sog_col] <= 0.5).sum():,}")
    print(f" Flagged >10% diff:      {df_final['flag_10_percent'].sum():,}")
    print(f" Flagged >30% diff:      {df_final['flag_30_percent'].sum():,}")
    print("="*50 + "\n")

# Example usage
output_path = output_dir / "AIS_2020_05_LANDFLAG.csv"
final_path = output_dir / "AIS_2020_05_FINAL.csv"

if __name__ == "__main__":

    calculate_sog(
    input_csv= output_path,
    output_csv=final_path
    )

