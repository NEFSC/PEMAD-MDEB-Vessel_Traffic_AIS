###############################################################
##   DOWNLOAD AIS DATA FROM MARINE CADASTRE. FILTER TO THE   ##
##    SNE BOUNDING BOX AND ORGANIZE INTO MONTHLY CSV FILES.  ##
##         FILTER FOR PROJECT VESSELS.                       ##
###############################################################

import pandas as pd
import concurrent.futures
import requests
#import zipfile
import zstandard as zstd
import io
import os
from pathlib import Path
from bs4 import BeautifulSoup 
import geopandas as gpd
import shapely.wkb
import shapely.geometry
import threading
import duckdb

# Create a "Lock"
parquet_lock = threading.Lock()

# --- CONFIGURATION ---
YEAR = 2025
TARGET_MONTHS = [f"{m:02d}" for m in range(1, 13)] 
# Updated URL to the new Azure Blob location
index_url = f"https://noaaocm.blob.core.windows.net/ais/csv2/csv{YEAR}/index.html"
base_download_url = f"https://noaaocm.blob.core.windows.net/ais/csv2/csv{YEAR}/"

project_path = Path("D:/")
home_directory = Path.home()
arcgis_path = home_directory / "Documents" / "ArcGIS" / "Projects" / "AIS_Download"
staging_dir = project_path / "AIS_Staging" / f"AIS_Staging_{YEAR}"
output_dir = project_path / f"AIS_Monthly_{YEAR}"
output_path = project_path
output_csv = project_path / "AIS" / f"Vessels_AIS_{YEAR}.csv"
# vessel_csv = project_path / "Vessels_SNE.csv"

output_dir.mkdir(parents=True, exist_ok=True)
staging_dir.mkdir(parents=True, exist_ok=True)

# --- LOAD BOUNDING BOX ---
gdb_path = arcgis_path / "BoundingBox.gdb"
layer_name = "SNE_BoundingBox"

try:
    gdf_bbox = gpd.read_file(gdb_path, layer=layer_name, engine="pyogrio")
    # Coordinates must be in Decimal Degrees (WGS84) to match AIS data
    # Marine Cadastre AIS is EPSG:4326
    if gdf_bbox.crs.to_epsg() != 4326:
        gdf_bbox = gdf_bbox.to_crs(epsg=4326)
    
    XMIN, YMIN, XMAX, YMAX = gdf_bbox.total_bounds
    print(f"Filtering with BBox: {XMIN}, {YMIN}, {XMAX}, {YMAX}")
except Exception as e:
    print(f"Error loading GDB: {e}")

# ---  PARALLEL DOWNLOAD & FILTER ---
def download_and_filter(url):
    filename = os.path.basename(url)
    
    # Filter by Month Name BEFORE downloading
    # Filename format: AIS_2018_01_01.zip
    parts = filename.split('-')
    if len(parts) < 3 or parts[2] not in TARGET_MONTHS:
        return None 

    parquet_name = filename.replace(".csv.zst", ".parquet")
    out_path = staging_dir / parquet_name
    
    if out_path.exists():
        return f"Already exists: {parquet_name}"

    try:
        r = requests.get(url, timeout=60, stream=True)
        print(f"Checking {filename}: Status {r.status_code}")
        r.raise_for_status() 

        # DECOMPRESS ZST STREAM
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(r.raw) as reader:
            # Wrap the reader in io.TextIOWrapper to make it readable by pandas
            # as a standard text/csv stream
            with io.TextIOWrapper(reader, encoding='utf-8') as text_stream:
                df = pd.read_csv(text_stream, low_memory=False)
                
                # --- DATA QUALITY FILTERS ---
                df['sog'] = pd.to_numeric(df['sog'], errors='coerce')
                df['mmsi'] = pd.to_numeric(df['mmsi'], errors='coerce')
                df = df.dropna(subset=['mmsi', 'sog'])
                df = df[(df['mmsi'] >= 100000000) & (df['mmsi'] <= 999999999)]
                df = df[df['sog'] <= 40]

                if 'vessel_name' in df.columns:
                    df['vessel_name'] = df['vessel_name'].astype(str).fillna("Unknown")
                if 'base_date_time' in df.columns:
                    df['base_date_time'] = pd.to_datetime(df['base_date_time'], errors='coerce')

                # BOUNDING BOX SPATIAL FILTER              
                mask = (df['latitude'] >= YMIN) & (df['latitude'] <= YMAX) & \
                       (df['longitude'] >= XMIN) & (df['longitude'] <= XMAX)
                
                filtered = df[mask].copy()
                
                if not filtered.empty:
                    filtered['geometry_wkb'] = [
                        shapely.wkb.dumps(shapely.geometry.Point(lon, lat)) 
                        for lon, lat in zip(filtered['longitude'], filtered['latitude'])
                    ]

                    with parquet_lock:
                        filtered.to_parquet(out_path, index=False, engine='pyarrow')
                    
                    return f"SAVED: {parquet_name} ({len(filtered):,} points)"
                return f"EMPTY: {parquet_name}"
                
    except Exception as e:
        return f"FAILED: {filename} - {e}"

def run_stage_1():
    print(f"--- Scraping {index_url} ---")
    response = requests.get(index_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # --- Robust URL Construction ---
    zst_links = []
    for l in soup.find_all('a'):
        href = l.get('href', '')
        if href.endswith('.csv.zst'):
            # Construct the full Azure URL
            clean_href = href.split('/')[-1] # Get just the filename
            full_url = f"{base_download_url}{clean_href}"
            zst_links.append(full_url)

    print(f"Found {len(zst_links)} daily files. Starting filter...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(download_and_filter, zst_links))
    
    for res in filter(None, results): 
        print(res)

def merge_daily_to_monthly():
    # Initialize DuckDB
    con = duckdb.connect()
    
    print(f"--- Starting Monthly Merge for {YEAR} ---")
    
    for month in TARGET_MONTHS:
        monthly_file = output_dir / f"AIS_{YEAR}_{month}.parquet"
        
        # Pattern to find all daily files for this specific month
        # Use the hyphen structure to match the new filenames
        daily_glob = f"ais-{YEAR}-{month}-*.parquet"
        daily_pattern = str(staging_dir / daily_glob)
        
        # Check if any files actually exist for this month before trying to merge
        daily_files_list = list(staging_dir.glob(daily_glob))
        
        if not daily_files_list:
            print(f"Skipping Month {month}: No daily files found.")
            continue
            
        print(f"Processing Month {month} ({len(daily_files_list)} files)...")
        
        try:
            # DuckDB SQL to combine all matching files and save to a new Parquet
            con.execute(f"""
                COPY (
                    SELECT * FROM read_parquet('{daily_pattern}', union_by_name=True)
                ) TO '{monthly_file}' (FORMAT 'PARQUET', COMPRESSION 'ZSTD');
            """)
            
            print(f"Successfully created: {monthly_file.name}")
            
            # --- CLEANUP ---
            # Now that the monthly file is safe, delete the daily staging files to save space
            # for f in daily_files_list:
            #     os.remove(f)
            # print(f"Cleaned up staging files for month {month}.")
            
        except Exception as e:
            print(f"ERROR merging month {month}: {e}")

def convert_monthly_parquet_to_csv():
    """
    Converts all Parquet files in a folder to CSV.
    Converts 'geometry_wkb' binary column to a readable WKT string.
    """
    # Initialize DuckDB with Spatial support
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    # Find all parquet files
    parquet_files = list(output_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {output_dir}")
        return

    print(f"--- Starting CSV Conversion of {len(parquet_files)} files ---")

    for p_file in parquet_files:
        csv_file = output_dir / p_file.name.replace(".parquet", ".csv")
        
        print(f"Converting {p_file.name} to CSV...")
        
        try:
            # SQL logic: 
            # 1. Exclude the raw binary column
            # 2. Convert that binary to a readable WKT string
            con.execute(f"""
                COPY (
                    SELECT 
                        * EXCLUDE (geometry_wkb), 
                        ST_AsText(ST_GeomFromWKB(geometry_wkb)) AS geometry
                    FROM read_parquet('{p_file}')
                ) TO '{csv_file}' (HEADER, DELIMITER ',');
            """)
            print(f"Successfully exported: {csv_file.name}")
            
        except Exception as e:
            print(f"Failed to convert {p_file.name}: {e}")

def filter_ais_by_project_vessels():
    """
    Filters a directory of AIS Parquet files based on a CSV of specific 
    MMSIs and their associated project start/end dates.
    """
    # Read Vessel/Project CSV into a Pandas DataFrame
    print(f"Reading project vessel list from: {vessel_csv}")
    project_vessels = pd.read_csv(vessel_csv)
    
    # Ensure Date columns are actual datetime objects for the filter
    project_vessels['Start'] = pd.to_datetime(project_vessels['Start'])
    project_vessels['End'] = pd.to_datetime(project_vessels['End'])
    
    # Initialize DuckDB with Spatial support
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    
    # Use a glob pattern to treat all monthly parquets as one table
    parquet_pattern = Path(output_dir) / "*.parquet"
    
    print("Starting Join and Filter (Parquet -> CSV)...")
    
    # The Power Query
    # It only keeps rows where the MMSI matches AND the date is within the window.
    query = f"""
        COPY (
            SELECT 
                main.* EXCLUDE (geometry_wkb),
                ST_AsText(ST_GeomFromWKB(main.geometry_wkb)) AS geometry
            FROM read_parquet('{parquet_pattern}') AS main
            JOIN project_vessels AS pv 
              ON main.MMSI = pv.MMSI
            WHERE CAST(main.BaseDateTime AS TIMESTAMP) >= pv.Start
              AND CAST(main.BaseDateTime AS TIMESTAMP) <= pv.End
        ) TO '{output_csv}' (HEADER, DELIMITER ',');
    """
    
    try:
        con.execute(query)
        print(f"Success! Filtered data saved to: {output_csv}")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    run_stage_1()
    #merge_daily_to_monthly()
    #convert_monthly_parquet_to_csv()
    #filter_ais_by_project_vessels()