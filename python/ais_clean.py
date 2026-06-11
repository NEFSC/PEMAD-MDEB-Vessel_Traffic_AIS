###########################################
##   AIS DATA CLEANING AND PROCESSING    ##
###########################################

# --- IMPORT LIBARRIES ---
import pandas as pd
import geopandas as gpd
import duckdb
import shapely
import requests
from pathlib import Path
import numpy as np
from haversine import haversine_vector, Unit
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
year = 2020
month = "05"
input_dir = Path(f"D:/AIS_Data/AIS_Monthly_{year}")
output_dir = Path("D:/AIS_Cleaned_Test")

output_dir.mkdir(parents=True, exist_ok=True)


### ----- 1. REMOVE GPS JITTER --------- ####

# Remove points where there is less than 1 minute elapsed time between consecutive points 
def remove_gps_jitter():
    print("\n=== STEP 1: REMOVE GPS JITTER ===")
    
    # Connect to DuckDB
    con = duckdb.connect()
    
    # Grab raw monthly parquet file
    parquet_files = list(input_dir.glob(f"AIS_{year}_{month}*.parquet"))
    
    for f_path in parquet_files:
        # Create a temporary output file for the jitter-cleaned data
        temp_output_path = output_dir / f"{f_path.stem}_JITTER_CLEANED.parquet"
        print(f"Filtering jitter from {f_path.name}...")
        
        # SQL query to calculate elapsed time and filter out < 60s gaps
        # Add column 'seconds_since_prev' to show elapsed time between consecutive points
        query = f"""
            -- First, get total count of original rows for stats
            CREATE TABLE total_count AS SELECT COUNT(*) as cnt FROM read_parquet('{str(f_path)}');
            
            -- Reconstruct tracks to compute elapsed time, then save the filtered result
            COPY (
                WITH calculated_intervals AS (
                    SELECT 
                        *,
                        -- Calculate seconds between current and previous point per MMSI
                        epoch(base_date_time) - epoch(LAG(base_date_time) OVER (
                            PARTITION BY mmsi 
                            ORDER BY base_date_time
                        )) AS seconds_since_prev
                    FROM read_parquet('{str(f_path)}')
                )
                SELECT * FROM calculated_intervals
                -- Keep the first point of a trip (NULL) or points >= 60 seconds apart
                WHERE seconds_since_prev IS NULL OR seconds_since_prev >= 60
            ) TO '{str(temp_output_path)}' (FORMAT 'PARQUET');
            
            -- Get total count of filtered rows for stats
            CREATE TABLE filtered_count AS SELECT COUNT(*) as cnt FROM read_parquet('{str(temp_output_path)}');
        """
        
        try:
            con.execute(query)
            
            # Fetch stats from DuckDB to print summary
            orig_total = con.execute("SELECT cnt FROM total_count").fetchone()[0]
            filt_total = con.execute("SELECT cnt FROM filtered_count").fetchone()[0]
            removed_rows = orig_total - filt_total
            pct_removed = (removed_rows / orig_total) * 100 if orig_total > 0 else 0
            
            print(f"  Original Rows:     {orig_total:,}")
            print(f"  Cleaned Rows:      {filt_total:,}")
            print(f"  Jitter Removed:    {removed_rows:,} ({pct_removed:.2f}%)")
            print(f"  Saved Cleaned File: {temp_output_path.name}")
            
            # Drop stats tables before next loop iteration
            con.execute("DROP TABLE total_count; DROP TABLE filtered_count;")
            
        except Exception as e:
            print(f"  Failed to remove jitter: {e}")

if __name__ == "__main__":
    # Run Step 1: Remove GPS Jitter 
    remove_gps_jitter()

### ----- 2. FLAG COASTAL POINTS (WITHIN 1NM OF LAND) ----- ###

# We are not interested in looking at inland points
# Creating a 'within 1nm of land' flag will allow us to filter out those points

# Land layer
# BOEM Renewable States REST URL
rest_url = "https://services7.arcgis.com/G5Ma95RzqJRPKsWL/arcgis/rest/services/BOEM_Renewable_States/FeatureServer/0/query"

# Study Area Bounding Box (to clip the land locally)
# Use the same bounding box that I used for AIS data download
BBOX = {"xmin": -74.05, "ymin": 39.50, "xmax": -68.78, "ymax": 41.69}

def flag_coastal_points():
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
    # Clip to bounding box
    bbox_geom = gpd.GeoSeries([shapely.geometry.box(BBOX['xmin'], BBOX['ymin'], BBOX['xmax'], BBOX['ymax'])], crs="EPSG:4326")
    
    # Validate geometry
    land_gdf['geometry'] = land_gdf['geometry'].make_valid()
    land_gdf = gpd.clip(land_gdf, bbox_geom)
    
    # Project to Albers Equal Area Conic for accurate metric buffering
    land_gdf = land_gdf.to_crs(epsg=5070)
    
    # Use modern shapely vectorized buffer 
    land_gdf['geometry'] = shapely.buffer(land_gdf['geometry'].values, 1852)
    
    # DISSOLVE OVERLAPS INTO A SINGLE MULTIPOLYGON ---
    print("Dissolving overlapping buffer zones...")
    dissolved_geom = shapely.unary_union(land_gdf['geometry'].values)
    
    # Recreate a GeoDataFrame with the single unified shape
    land_gdf = gpd.GeoDataFrame(geometry=[dissolved_geom], crs="EPSG:5070")
    # Validate geometry and project back to WGS84
    land_gdf['geometry'] = land_gdf['geometry'].make_valid()
    land_gdf = land_gdf[~land_gdf.is_empty].to_crs(epsg=4326)
    
    # Convert to WKB for DuckDB
    land_gdf['geom_wkb'] = land_gdf['geometry'].apply(lambda x: x.wkb)
    land_df_for_duck = pd.DataFrame(land_gdf.drop(columns='geometry'))

    # 3. DUCKDB SPATIAL JOIN
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.register("buffered_land", land_df_for_duck)

    parquet_files = list(output_dir.glob(f"AIS_{year}_{month}*_JITTER_CLEANED.parquet"))

    for f_path in parquet_files:
        # Save output as CSV
        output_path = output_dir / f"{f_path.stem.replace('_JITTER_CLEANED', '')}_LANDFLAG.csv"
        print(f"Processing {f_path.name} -> {output_path.name}...")

        # DuckDB SQL query to flag rows on land
        query = f"""
            COPY (
                SELECT 
                    main.* EXCLUDE (geometry_wkb),
                    ST_AsText(ST_GeomFromWKB(main.geometry_wkb)) AS geometry,
                    (l.geom_wkb IS NOT NULL) AS flag_1nm_land,
                    CASE WHEN l.geom_wkb IS NOT NULL THEN 'Coastal point - within 1 nm of land' ELSE NULL END AS cleaning_notes
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
            
            # --- STATS ---
            # Load only the necessary columns to save memory
            df_check = pd.read_csv(output_path, usecols=['geometry', 'flag_1nm_land'])
            
            flagged_count = df_check['flag_1nm_land'].sum()
            total_rows = len(df_check)
            
            print(f"  Total Rows: {total_rows:,}")
            print(f"  Flagged (On Land): {flagged_count:,} ({(flagged_count/total_rows)*100:.2f}%)")

        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == "__main__":
    # Run Step 2: Flag coastal points 
    flag_coastal_points()

### ----- 3. FLAG/REMOVE POINTS WHERE CALCULATED SOG DIFFERS FROM REPORTED SOG ----- ###

# Stats function
def print_vessel_stats(df):
    # Print vessel summary, filtering out records where speed over ground is <1
    print("\n" + "="*80)
    print("VESSEL TYPE PERFORMANCE SUMMARY (Smoothed vs. Reported)")
    print("="*80)

    # Filter out records where reported SOG is 1 or lower
    moving_vessels_df = df.loc[df['sog'] > 1]

    # Print stats for all vessels
    stats= moving_vessels_df.agg({
        'sog': ['median', 'mean', 'std'],
        'moving_avg_sog': ['median', 'mean', 'std']
    })

    # Round for readability (2 decimal places)
    stats = stats.round(2)

    print(stats.to_string())
    print("="*80 + "\n")

    # Group by vessel type and calculate stats for both speed columns
    # 'sog' is the reported value, 'moving_avg_sog' is the 2-point moving average
    stats_grouped = moving_vessels_df.groupby('vessel_group').agg({
        'sog': ['median', 'mean', 'std'],
        'moving_avg_sog': ['median', 'mean', 'std']
    })

    # Round for readability (2 decimal places)
    stats_grouped = stats_grouped.round(2)

    # Flatten the multi-index columns for a cleaner print
    stats_grouped.columns = [
        'Rep_SOG_Median', 'Rep_SOG_Mean', 'Rep_SOG_Std',
        'Calc_SOG_Median', 'Calc_SOG_Mean', 'Calc_SOG_Std'
    ]

    print(stats_grouped.to_string())
    print("="*80 + "\n")

    # Print vessel summary, filtering out slow speeds and coastal pings
    print("\n" + "="*80)
    print("VESSEL TYPE PERFORMANCE SUMMARY (OCEAN-BASED) (Smoothed vs. Reported)")
    print("="*80)

    # Select records where reported SOG is > 1 AND pings are not flagged as within 1nm of land
    moving_ocean_vessels_df = df.loc[(df['sog'] > 1) & (~df['flag_1nm_land'])]

    # Print stats for all vessels
    stats_ocean= moving_ocean_vessels_df.agg({
        'sog': ['median', 'mean', 'std'],
        'moving_avg_sog': ['median', 'mean', 'std']
    })

    # Round for readability (2 decimal places)
    stats_ocean = stats_ocean.round(2)

    print(stats_ocean.to_string())
    print("="*80 + "\n")

    # Group by vessel type and calculate stats for both speed columns
    # 'sog' is the reported value, 'moving_avg_sog' is the 2-point moving average
    stats_grouped_ocean = moving_ocean_vessels_df.groupby('vessel_group').agg({
        'sog': ['median', 'mean', 'std'],
        'moving_avg_sog': ['median', 'mean', 'std']
    })

    # Round for readability (2 decimal places)
    stats_grouped_ocean= stats_grouped_ocean.round(2)

    # Flatten the multi-index columns for a cleaner print
    stats_grouped_ocean.columns = [
        'Rep_SOG_Median', 'Rep_SOG_Mean', 'Rep_SOG_Std',
        'Calc_SOG_Median', 'Calc_SOG_Mean', 'Calc_SOG_Std'
    ]

    print(stats_grouped_ocean.to_string())
    print("="*80 + "\n")

# Calculating SOG outliers function
def remove_speed_mismatch_outliers(df):
    print("\n" + "="*50)
    print(" --- OUTLIER DETECTION & SIGNED ANOMALY BOXPLOT (SOG > 1) ---")
    print("="*50)
    
    total_rows = len(df)
    if total_rows == 0:
        print("Error: Input DataFrame is empty.")
        return df

    # 1. ISOLATE MOVING VESSELS & CALCULATE SIGNED AND ABSOLUTE ERRORS
    moving_mask = df['sog'] > 1
    
    # Signed anomaly (keeps direction: positive means calculated is faster, negative means reported is faster)
    moving_errors_signed = df.loc[moving_mask, 'moving_avg_sog'] - df.loc[moving_mask, 'sog']
    
    # Absolute error (still required to calculate standard statistical upper fence)
    moving_errors_abs = moving_errors_signed.abs()
    moving_rows_count = len(moving_errors_abs)
    
    if moving_rows_count == 0:
        print("Warning: No moving vessels (SOG > 1) found to evaluate.")
        return df

    # 2. STATS METHOD: INTERQUARTILE RANGE (IQR) ON ABSOLUTE ERRORS
    q1 = moving_errors_abs.quantile(0.25)
    q3 = moving_errors_abs.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + (1.5 * iqr)
    
    # Identify outliers where the absolute magnitude exceeds the fence
    iqr_outlier_mask = pd.Series(False, index=df.index)
    iqr_outlier_mask.loc[moving_mask] = moving_errors_abs > upper_fence
    
    iqr_removed_count = iqr_outlier_mask.sum()
    iqr_pct_of_moving = (iqr_removed_count / moving_rows_count) * 100

    print(f" Moving Rows Evaluated (>1kt): {moving_rows_count:,}")
    print(f" Moving IQR Upper Fence:       {upper_fence:.4f} knots (Absolute magnitude)")
    print(f" Outliers Detected & Removed:  {iqr_removed_count:,} rows ({iqr_pct_of_moving:.2f}%)")

    # 3. PREPARE DATA FOR THE SIGNED BEFORE/AFTER BOXPLOT
    try:
        print("Displaying Before vs. After outlier removal signed boxplot...")
        
        # Signed errors before any filtering
        df_before = pd.DataFrame({
            'Error': moving_errors_signed,
            'Timeline': 'Before Outlier Removal'
        })
        
        # Signed errors after filtering (only keeping rows where absolute error <= upper_fence)
        df_after = pd.DataFrame({
            'Error': moving_errors_signed[moving_errors_abs <= upper_fence],
            'Timeline': 'After Outlier Removal'
        })

        # 4. PLOT LAYOUT
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
        sns.set_theme(style="whitegrid")

        # Plot 1: Before Removal (Left Axis)
        sns.boxplot(
            data=df_before, 
            y='Error', 
            ax=axes[0], 
            color='#e74c3c', # Red
            width=0.4
        )
        axes[0].axhline(0, color='black', linestyle='--', alpha=0.7) # Reference line at 0 error
        axes[0].set_title("Before Outlier Removal\n(Full Signed Anomalies)", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Signed Speed Error (Calculated - Reported in Knots)", fontsize=11)
        axes[0].set_xticklabels(["Moving Vessels (SOG > 1)"])

        # Plot 2: After Removal (Right Axis)
        sns.boxplot(
            data=df_after, 
            y='Error', 
            ax=axes[1], 
            color='#2ecc71', # Green
            width=0.4
        )
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.7) # Reference line at 0 error
        axes[1].set_title(f"After Outlier Removal\n(Cleaned Two-Tailed Range)", fontsize=12, fontweight='bold')
        axes[1].set_ylabel("Signed Speed Error (Calculated - Reported in Knots)", fontsize=11)
        axes[1].set_xticklabels(["Cleaned Vessels"])

        # Title formatting
        plt.suptitle(f"AIS Signed Speed Anomaly Distribution Profile ({year}-{month})", fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        plt.show()
        plt.close()

    except Exception as chart_err:
        print(f"  Warning: Failed to render boxplot. Details: {chart_err}")

    # 5. FILTER AND RETURN CLEANED DATA
    df_cleaned = df[~iqr_outlier_mask].copy()
    return df_cleaned

# Function to calculate speed over ground
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

    # 3. USE ELAPSED TIME FROM STEP 1
    # Convert pre-calculated 'seconds_since_prev' column into hours
    df['time_hours'] = df['seconds_since_prev'] / 3600

    # Drop the first point of each trip (which has NaN/Null time hours)
    df = df[df['time_hours'].notna()].copy()

    # 4. CALCULATE SHIFTS & DISTANCE (Vectorized)
    df['prev_lat'] = df.groupby(mmsi_col)[lat_col].shift(1)
    df['prev_lon'] = df.groupby(mmsi_col)[lon_col].shift(1)
    
    points_now = df[[lat_col, lon_col]].to_numpy()
    points_prev = df[['prev_lat', 'prev_lon']].to_numpy()
    # Calculate distance between two points using haversine python package
    df['distance_nm'] = haversine_vector(points_now, points_prev, unit=Unit.NAUTICAL_MILES)

    # 5. CALCULATE RAW & AVERAGED SOG
    df['raw_calc_sog'] = df['distance_nm'] / df['time_hours']

    # --- REMOVE STATE-SPEED MISMATCHES (SOG CONTRADICTIONS) ---
    # Condition 1: Calculated speed says it's moving fast, reported speed (GPS) says it's stopped/anchored
    cond_calculated_fast_reported_stopped = (df['raw_calc_sog'] > 2.0) & (df[reported_sog_col] <= 1.0)
    
    # Condition 2: Reported speed (GPS) says it's moving fast, calculated distance says it's stationary
    cond_reported_fast_calculated_stopped = (df[reported_sog_col] > 2.0) & (df['raw_calc_sog'] <= 1.0)
    
    # Combine masks
    mismatch_mask = cond_calculated_fast_reported_stopped | cond_reported_fast_calculated_stopped
    
    # Track stats for printing
    count_calc_anoms = cond_calculated_fast_reported_stopped.sum()
    count_rep_anoms = cond_reported_fast_calculated_stopped.sum()
    
    # Drop the anomalies
    df = df[~mismatch_mask].copy()
    # -----------------------------------------------------------------

    # --- FILTER OUT EXTREME SPEED ANOMALIES (> 40 KNOTS) ---
    # Remove rows where calculated speed is greater than 40 knots (impossibly fast = GPS positional error)
    # Store length before filtering
    count_before_speed_filter = len(df)
    
    # Apply the speed threshold filter
    df = df[df['raw_calc_sog'] <= 40].copy()
    
    # Calculate how many rows were dropped
    extreme_speed_removed = count_before_speed_filter - len(df)
    # -------------------------------------------------------

    # 2-point Moving Average (The "Smoother")
    # This runs on data that has already had pings > 40 knots removed
    df['moving_avg_sog'] = df.groupby(mmsi_col)['raw_calc_sog'].transform(
        lambda x: x.rolling(window=2, min_periods=1).mean()
    )

    # 6. CLEANUP & INITIAL EXPORT DROP
    # Drop positional shifts helper columns to keep CSV clean before statistical step
    cols_to_drop = ['prev_lat', 'prev_lon', 'time_hours', 'distance_nm']
    df_working = df.drop(columns=cols_to_drop)

    # 7. RUN OUTLIER REMOVAL (Drops the extreme IQR errors first)
    df_final = remove_speed_mismatch_outliers(df_working)

    # 8. FLAG MINOR ANOMALIES (Runs only on the surviving, clean rows!)
    valid_mask = df_final[reported_sog_col] > 1
    land_mask = df_final['flag_1nm_land'] == True  
    slow_mask = df_final[reported_sog_col] <= 1
    land_and_slow_count = (land_mask & slow_mask).sum()

    # Initialize flag columns on the clean dataframe
    df_final['percent_difference'] = np.nan
    df_final['flag_10_percent'] = False
    df_final['flag_30_percent'] = False

    # Calculate difference using the moving average on surviving rows
    df_final.loc[valid_mask, 'percent_difference'] = (
        (abs(df_final.loc[valid_mask, 'moving_avg_sog'] - df_final.loc[valid_mask, reported_sog_col])) / 
        (df_final.loc[valid_mask, reported_sog_col] + 0.001)
    ) * 100

    # Set Flags
    df_final.loc[valid_mask, 'flag_10_percent'] = df_final.loc[valid_mask, 'percent_difference'] > 10
    df_final.loc[valid_mask, 'flag_30_percent'] = df_final.loc[valid_mask, 'percent_difference'] > 30
    
    # Write the truly clean and accurately flagged data to file
    df_final.to_csv(output_csv, index=False)

    # SUMMARY PRINT
    print("\n" + "="*50)
    print(" --------------- AIS SOG ANALYSIS ----------------------")
    print(f" Total input points:          {total_original_points:,}")
    print(f" Processed points:            {len(df_final):,}")
    print(f" Mismatch (Calc >2, Rep <=1): {count_calc_anoms:,} points removed")
    print(f" Mismatch (Rep >2, Calc <=1): {count_rep_anoms:,} points removed")
    print(f" Extreme Speed (>40kts):      {extreme_speed_removed:,} points removed") 
    print(f" Ignored (SOG <= 1):          {(df_final[reported_sog_col] <= 1).sum():,}") # <-- Changed to df_final
    print(f" Flagged >10% diff:           {df_final['flag_10_percent'].sum():,}")
    print(f" Flagged >30% diff:           {df_final['flag_30_percent'].sum():,}")
    print(f" Near Land AND Slow:          {land_and_slow_count:,}")
    print("="*50 + "\n")

    print_vessel_stats(df_final)

# Usage
output_path = output_dir / "AIS_2020_05_LANDFLAG.csv"
final_path = output_dir / "AIS_2020_05_FINAL.csv"

if __name__ == "__main__":
    # Step 3: Calculate SOG and flag/remove outliers
    calculate_sog(
    input_csv= output_path,
    output_csv=final_path
    )

### ----- 4. FLAG/REMOVE POINTS USING HEADING CHANGE ANALYSIS ----- ###
# 



