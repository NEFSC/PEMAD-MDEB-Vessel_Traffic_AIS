###############################################################
##   DOWNLOAD AIS DATA FROM MARINE CADASTRE. FILTER TO THE   ##
##    SNE BOUNDING BOX AND ORGANIZE INTO MONTHLY CSV FILES.  ##
##         FILTER FOR PROJECT VESSELS.                       ##
###############################################################

import pandas as pd
import concurrent.futures
import requests
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

# Create a "Lock" for thread-safe Parquet writing
parquet_lock = threading.Lock()

# --- GLOBAL CONFIGURATION ---
YEARS_TO_DOWNLOAD = range(2015, 2026) # 2015 through 2025
TARGET_MONTHS = [f"{m:02d}" for m in range(1, 13)]

# Path to where data will be saved
project_path = Path("D:/")

# Find Bounding Box fgd in current directory
current_dir = Path.cwd()
project_root = None
for parent in [current_dir] + list(current_dir.parents):
    if (parent / "data").is_dir() and (parent / "python").is_dir():
        project_root = parent
        break

# 3. Fallback just in case it's running somewhere weird
if not project_root:
    project_root = current_dir

# --- LOAD BOUNDING BOX ---
gdb_path = project_root / "data" / "BoundingBox.gdb"
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
    exit()

# ---  PARALLEL DOWNLOAD & FILTER ---
def download_and_filter(url, current_staging_dir):
    filename = os.path.basename(url)
    
    # Filename format: ais-2024-01-01.csv.zst
    parts = filename.split('-')
    if len(parts) < 3 or parts[2] not in TARGET_MONTHS:
        return None 

    parquet_name = filename.replace(".csv.zst", ".parquet")
    out_path = current_staging_dir / parquet_name
    
    if out_path.exists():
        return f"Already exists: {parquet_name}"

    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status() 

        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(r.raw) as reader:
            with io.TextIOWrapper(reader, encoding='utf-8') as text_stream:
                df = pd.read_csv(text_stream, low_memory=False)
                
                # --- NORMALIZE COLUMN NAMES ---
                # This handles shifts between lowercase/uppercase across years
                df.columns = [c.lower() for c in df.columns]

                # --- DATA QUALITY FILTERS ---
                df['sog'] = pd.to_numeric(df['sog'], errors='coerce')
                df['mmsi'] = pd.to_numeric(df['mmsi'], errors='coerce')
                # Remove rows where MMSI or SOG is NA
                df = df.dropna(subset=['mmsi', 'sog'])
                # Remove rows where the MMSI number is not 9 digits (all vessel MMSI should be 9 digits, bad data)
                df = df[(df['mmsi'] >= 100000000) & (df['mmsi'] <= 999999999)]
                # Remove rows where the SOG >40 (impossible speed, bad data)
                df = df[df['sog'] <= 40]

                if 'vessel_name' in df.columns:
                    df['vessel_name'] = df['vessel_name'].astype(str).fillna("Unknown")
                if 'base_date_time' in df.columns:
                    df['base_date_time'] = pd.to_datetime(df['base_date_time'], errors='coerce')

                # BOUNDING BOX SPATIAL FILTER   
                # Only grab AIS data within the bounding box           
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
                    # Save as parquet file 
                    return f"SAVED: {parquet_name} ({len(filtered):,} points)"
                return f"EMPTY: {parquet_name}"
                
    except Exception as e:
        return f"FAILED: {filename} - {e}"

def process_year(year):
    print(f"\n======= STARTING YEAR {year} =======")
    
    # 1. SETUP DIRECTORIES
    staging_dir = project_path / "AIS_Staging" / f"AIS_Staging_{year}"
    output_dir = project_path / f"AIS_Monthly_{year}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. SCRAPE LINKS
    index_url = f"https://noaaocm.blob.core.windows.net/ais/csv2/csv{year}/index.html"
    base_download_url = f"https://noaaocm.blob.core.windows.net/ais/csv2/csv{year}/"
    
    try:
        response = requests.get(index_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        zst_links = []
        for l in soup.find_all('a'):
            href = l.get('href', '')
            if href.endswith('.csv.zst'):
                clean_href = href.split('/')[-1]
                zst_links.append(f"{base_download_url}{clean_href}")

        print(f"Found {len(zst_links)} daily files for {year}.")

        # 3. DOWNLOAD IN PARALLEL
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Use a helper to pass both the URL and the year-specific staging dir
            results = list(executor.map(lambda u: download_and_filter(u, staging_dir), zst_links))
        
        for res in filter(None, results): 
            print(res)

    except Exception as e:
        print(f"Error processing year {year}: {e}")

# --- RUN ---
if __name__ == "__main__":
    for year in YEARS_TO_DOWNLOAD:
        process_year(year)

# --- CONVERT DAILY FILES TO MONTHLY FILES
def merge_daily_to_monthly_multiyear(years_list):
    # Initialize DuckDB
    con = duckdb.connect()
    
    for year in years_list:
        print(f"\n--- Starting Monthly Merge for {year} ---")
        
        # DYNAMIC DIRECTORY PATHS
        # These must match the folder names created during the download stage
        current_staging_dir = project_path / "AIS_Staging" / f"AIS_Staging_{year}"
        current_output_dir = project_path / f"AIS_Monthly_{year}"
        
        # Ensure output directory exists
        current_output_dir.mkdir(parents=True, exist_ok=True)

        for month in TARGET_MONTHS:
            monthly_file = current_output_dir / f"AIS_{year}_{month}.parquet"
            
            # Pattern to find all daily files for this specific year/month
            daily_glob = f"ais-{year}-{month}-*.parquet"
            daily_pattern = str(current_staging_dir / daily_glob)
            
            # Check if any files actually exist for this month
            daily_files_list = list(current_staging_dir.glob(daily_glob))
            
            if not daily_files_list:
                print(f"Skipping {year}-{month}: No daily files found in {current_staging_dir}")
                continue
                
            if monthly_file.exists():
                print(f"Skipping {year}-{month}: Monthly file already exists.")
                continue

            print(f"Processing {year}-{month} ({len(daily_files_list)} files)...")
            
            try:
                # union_by_name=True is critical here because schemas change over 10 years!
                con.execute(f"""
                    COPY (
                        SELECT * FROM read_parquet('{daily_pattern}', union_by_name=True)
                    ) TO '{monthly_file}' (FORMAT 'PARQUET', COMPRESSION 'ZSTD');
                """)
                
                print(f"Successfully created: {monthly_file.name}")
                
                ## --- OPTIONAL CLEANUP ---
                ## Remove daily files 
                # for f in daily_files_list:
                #     os.remove(f)
                
            except Exception as e:
                print(f"ERROR merging {year}-{month}: {e}")

# --- RUN ---
if __name__ == "__main__":
    # Pass a year range or a specific list
    target_years = range(2015, 2026) 
    merge_daily_to_monthly_multiyear(target_years)

# --- CONVERT MONTHLY PARQUET TO MONTHLY CSV
# I did it this way because it was much faster to first create parquet files, then convert them to csv
# as opposed to only making csv files
# Utlizing DuckDB to create parquet files is very fast for download/filtering 
def convert_monthly_parquet_to_csv_multiyear(years_list):
    """
    Iterates through multiple years, finds monthly Parquet files, 
    and converts them to CSV with readable WKT geometry.
    """
    # Initialize DuckDB with Spatial support
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    for year in years_list:
        # Define the directory for the specific year
        current_output_dir = project_path / f"AIS_Monthly_{year}"
        
        if not current_output_dir.exists():
            print(f"Skipping Year {year}: Directory {current_output_dir} does not exist.")
            continue

        # Find all parquet files for this year
        parquet_files = list(current_output_dir.glob("*.parquet"))
        
        if not parquet_files:
            print(f"No parquet files found for {year}.")
            continue

        print(f"\n--- Starting CSV Conversion for {year} ({len(parquet_files)} files) ---")

        for p_file in parquet_files:
            csv_file = current_output_dir / p_file.name.replace(".parquet", ".csv")
            
            # Skip if CSV already exists to save time on re-runs
            if csv_file.exists():
                print(f"  Skipping {p_file.name}: CSV already exists.")
                continue

            print(f"  Converting {p_file.name}...")
            
            try:
                # SQL logic: 
                # 1. EXCLUDE binary column and any leftover "junk" columns from previous joins
                # 2. Convert WKB binary to readable WKT (Well-Known Text)
                con.execute(f"""
                    COPY (
                        SELECT 
                            * EXCLUDE (
                                geometry_wkb
                            ), 
                            ST_AsText(ST_GeomFromWKB(geometry_wkb)) AS geometry
                        FROM read_parquet('{p_file}')
                    ) TO '{csv_file}' (HEADER, DELIMITER ',');
                """)
                print(f"    Successfully exported: {csv_file.name}")
                
            except Exception as e:
                print(f"    Failed to convert {p_file.name}: {e}")

# --- RUN ---
if __name__ == "__main__":
    # Define years 2015-2025
    target_years = range(2015, 2026)
    convert_monthly_parquet_to_csv_multiyear(target_years)

# ---- ADD VESSEL TYPE GROUPS ---
# Use the vessel type codes csv to map vessel type groups to the finalized datasets
# Add a column vessel group to all the final data

data_dir = Path("D:/AIS_Monthly_2025")
mapping_csv = project_root / "data" / "vesseltypecodes.csv"

def add_vessel_groups_recursive():
    con = duckdb.connect()

    print("--- Loading Mapping Table ---")
    # Load the mapping CSV and ensure 'Vessel_Type' is treated as an integer
    con.execute(f"""
        CREATE TABLE vessel_mapping AS 
        SELECT DISTINCT 
            CAST(Vessel_Type AS INTEGER) AS Vessel_Type, 
            Vessel_Group 
        FROM read_csv_auto('{mapping_csv}')
    """)

    # Find all files recursively
    all_files = list(data_dir.glob("**/*.parquet")) + list(data_dir.glob("**/*.csv"))
    print(f"Found {len(all_files)} files. Starting processing...")

    for file_path in all_files:
        if file_path.name.startswith("temp_"):
            continue
            
        print(f"Processing: {file_path.relative_to(data_dir)}")
        
        ext = file_path.suffix.lower()
        temp_output = file_path.with_name(f"temp_{file_path.name}")

        try:
            # Look at the headers of the current AIS file
            if ext == '.parquet':
                cols_query = f"DESCRIBE SELECT * FROM read_parquet('{file_path}')"
            else:
                cols_query = f"DESCRIBE SELECT * FROM read_csv_auto('{file_path}')"
            
            actual_cols = [c[0] for c in con.execute(cols_query).fetchall()]

            # Identify which "Type" column exists in this specific year's data
            # There are multiple versions for different years
            type_col = None
            for variant in ['VesselType', 'vessel_type', 'Vessel_Type', 'vesseltype']:
                if variant in actual_cols:
                    type_col = variant
                    break
            
            if not type_col:
                print(f"  SKIPPING: No vessel type column found in {file_path.name}")
                continue

            # Build the Join Query
            # Use EXCLUDE (join_key) so the temporary column doesn't save to the file
            source_func = f"read_parquet('{file_path}')" if ext == '.parquet' else f"read_csv_auto('{file_path}')"
            
            query = f"""
                SELECT 
                    main.* EXCLUDE (
                        join_key, 
                    ), 
                    map.Vessel_Group 
                FROM (
                    SELECT 
                        *, 
                        TRY_CAST("{type_col}" AS INTEGER) AS join_key
                    FROM {source_func}
                ) AS main
                LEFT JOIN vessel_mapping AS map 
                    ON main.join_key = map.Vessel_Type
            """

            # Execute the export
            if ext == '.parquet':
                con.execute(f"COPY ({query}) TO '{temp_output}' (FORMAT 'PARQUET', COMPRESSION 'ZSTD')")
            else:
                con.execute(f"COPY ({query}) TO '{temp_output}' (FORMAT 'CSV', HEADER)")

            # Atomic Swap (Replace original with updated version)
            file_path.unlink() 
            temp_output.rename(file_path)
            
        except Exception as e:
            print(f"  ERROR on {file_path.name}: {e}")
            if temp_output.exists():
                temp_output.unlink()

    print("--- Processing Complete ---")

if __name__ == "__main__":
    add_vessel_groups_recursive()


# --- FILTER FOR CONSTRUCTION RELATED VESSELS IN DATA ---
# --- CONFIGURATION ---
YEARS = range(2017, 2026)  # 2017 to 2025
TARGET_MONTHS = [f"{m:02d}" for m in range(1, 13)]
vessel_csv = project_root / "data" / "OSW_Vessels_MMSI.csv"
construction_output_base = project_path / "AIS_ConstructionVessels"

def filter_ais_by_project_vessels_multiyear():
    """
    Loops through years and months to filter AIS Parquet files
    by project vessel MMSIs and dates, outputting monthly CSVs.
    """
    # 1. Prepare Project Vessel Data
    print(f"Reading project vessel list from: {vessel_csv}")
    project_vessels = pd.read_csv(vessel_csv)
    project_vessels['Start'] = pd.to_datetime(project_vessels['Start'])
    project_vessels['End'] = pd.to_datetime(project_vessels['End'])
    
    # 2. Initialize DuckDB
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    # Register the dataframe once
    con.register('project_vessels', project_vessels)

    # 3. Create Output Parent Directory
    construction_output_base.mkdir(parents=True, exist_ok=True)

    # 4. Nested Loop: Year -> Month
    for year in YEARS:
        # Path to where monthly parquets are stored
        input_dir = project_path / f"AIS_Monthly_{year}"
        
        if not input_dir.exists():
            print(f"Skipping Year {year}: Directory not found.")
            continue

        for month in TARGET_MONTHS:
            # Define input and output file paths
            # Matches format: AIS_2021_06.parquet
            p_file = input_dir / f"AIS_{year}_{month}.parquet"
            output_csv = construction_output_base / f"Vessels_AIS_{year}_{month}.csv"

            if not p_file.exists():
                continue # Skip months that don't have data

            if output_csv.exists():
                print(f"  Skipping {year}-{month}: CSV already exists.")
                continue

            print(f"Filtering {year}-{month} for project vessels...")

            # --- THE QUERY ---
            # Find all AIS pings in data where MMSI and start and end date match 
            # the OSW_Vessels_MMSI csv
            query = f"""
                COPY (
                    SELECT 
                        main.* EXCLUDE (geometry_wkb),
                        ST_AsText(ST_GeomFromWKB(main.geometry_wkb)) AS geometry
                    FROM read_parquet('{p_file}', union_by_name=True) AS main
                    JOIN project_vessels AS pv 
                      ON main.mmsi = pv.MMSI
                    WHERE CAST(main.base_date_time AS TIMESTAMP) >= pv.Start
                      AND CAST(main.base_date_time AS TIMESTAMP) <= pv.End
                ) TO '{output_csv}' (HEADER, DELIMITER ',');
            """

            try:
                con.execute(query)
                # Check if the file was created and has data (header + rows)
                if output_csv.stat().st_size < 100: 
                     print(f"    Note: No matching vessels found for {year}-{month}")
            except Exception as e:
                print(f"    Error processing {year}-{month}: {e}")

if __name__ == "__main__":
    filter_ais_by_project_vessels_multiyear()


# --- CREATE YEARLY CONSTRUCTION RELATED VESSEL FILE GEODATABASES ---

# --- CONFIGURATION ---
input_base_dir = Path("D:/AIS_ConstructionVessels")
gdb_output_dir = Path("D:/AIS_Geodatabases")
gdb_output_dir.mkdir(parents=True, exist_ok=True)

YEARS = range(2025, 2026)

def create_yearly_gdbs():
    print("Starting Geodatabase Creation...")

    for year in YEARS:
        # Define GDB path
        gdb_path = gdb_output_dir / f"Vessels_AIS_{year}.gdb"
        
        # Find all CSVs for this year
        csv_files = list(input_base_dir.glob(f"Vessels_AIS_{year}_*.csv"))

        if not csv_files:
            continue

        print(f"\n--- Processing Year {year} ({len(csv_files)} files) ---")

        for csv_path in csv_files:
            # Create a layer name (e.g., Month_01)
            # One layer per month in each yearly file geodatabase 
            month_val = csv_path.stem.split('_')[-1]
            layer_name = f"Vessels_AIS_{year}_{month_val}"

            print(f"  Converting {csv_path.name} to GDB Layer...")

            try:
                # 1. Load CSV
                df = pd.read_csv(csv_path)

                if df.empty:
                    print(f"Note: {csv_path.name} is empty. Skipping.")
                    continue

                # 2. Convert WKT string to actual Geometry objects
                df['geometry'] = df['geometry'].apply(shapely.wkb.loads if 'wkb' in csv_path.name else shapely.wkt.loads)
                
                # 3. Create GeoDataFrame
                # AIS data is WGS84 (EPSG:4326)
                gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

                # 4. Write to File Geodatabase using pyogrio
                gdf.to_file(
                    str(gdb_path), 
                    layer=layer_name, 
                    driver="OpenFileGDB", 
                    engine="pyogrio"
                )
                print(f"Successfully added {layer_name} to {gdb_path.name}")

            except Exception as e:
                print(f"Error processing {csv_path.name}: {e}")

if __name__ == "__main__":
    create_yearly_gdbs()
