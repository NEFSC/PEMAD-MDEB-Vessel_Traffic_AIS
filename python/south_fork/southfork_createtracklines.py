########################################################
##     CREATE TRIP TRACKLINES FROM AIS DATA           ##
##    WITH STATUS TRANSITIONS AND STATIONARY FLAGS    ##
########################################################

import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import LineString

# CONFIGURATION
# Set folder paths
base_path = Path.cwd()
gdb_path = base_path / "data" / "south_fork" / "south_fork_vessel_ais.gdb"
input_layer = "south_fork_vessel_merged"
ports_layer = "south_fork_ports"
output_layer = "south_fork_vessel_trips_lines"
line_gdb = base_path / "data" / "south_fork" / "south_fork_vessel_trips.gdb"

# State Submerged Land URL
submergedlands_url = "https://coast.noaa.gov/arcgis/rest/services/Hosted/USStateSubmergedLands/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson"

# Port Coordinates from South Fork Construction Report
# Used to create a Port Point Layer for mapping/visualization
ports_data = {
    'ProvPort': (-71.391, 41.802), 'New_London': (-72.091, 41.354),
    'Point_Judith': (-71.511, 41.379), 'Quonset': (-71.415, 41.585),
    'Montauk': (-71.937, 41.074), 'New_Bedford': (-70.921, 41.6345),
    'Bridgeport': (-73.181, 41.173), 'Shinnecock': (-72.476, 40.842),
    'Fall_River': (-71.164, 41.704), 'Fairhaven': (-70.908, 41.624),
    'Newport_RI': (-71.328, 41.484), 'Sakonnet_Harbor': (-71.193, 41.464),
    'Brooklyn_NY': (-74.015, 40.672), 'Charleston_SC': (-79.925, 32.773),
    'Corpus_Christi': (-97.388, 27.793) #, 'Millville_NJ': (-75.044, 39.213) # Millville, NJ is listed in the report but is not coastal
}

# Convert dictionary to a list of records
ports_list = [{'port_name': name, 'lon': coords[0], 'lat': coords[1]} 
             for name, coords in ports_data.items()]

# Create a standard Pandas DataFrame
ports_df = pd.DataFrame(ports_list)

# Convert to a GeoDataFrame
ports_gdf = gpd.GeoDataFrame(
    ports_df, 
    geometry=gpd.points_from_xy(ports_df.lon, ports_df.lat),
    crs="EPSG:4326"  # Standard WGS84 coordinate system
)

# Export to the file geodatabase 
ports_gdf.to_file(str(gdb_path), layer=ports_layer, driver="OpenFileGDB", engine="pyogrio")
print(f"--- Port point feature class created at {gdb_path} ---")


# Inside your run_trackline_pipeline() function:
print("Fetching submerged lands boundary polygons...")
submergedlands_url=(submergedlands_url)

# Function to create tracklines from AIS points
def run_trackline_pipeline():
    # Load the merged point layer
    print(f"Reading points from {input_layer}...")
    gdf = gpd.read_file(str(gdb_path), layer=input_layer, engine="pyogrio")
    gdf.columns = gdf.columns.str.upper()
    gdf['BASEDATETIME'] = pd.to_datetime(gdf['BASEDATETIME'])
    # Sort points by MMSI and time
    gdf = gdf.sort_values(['MMSI', 'BASEDATETIME'])
    # Ensure active geometry is set
    gdf = gdf.set_geometry('GEOMETRY')
    
    # Fetch submerged lands boundary polygon
    print("Fetching submerged lands boundary lines and building spatial index...")
    print("Fetching submerged lands boundary polygons...")
    state_waters_gdf = gpd.read_file(submergedlands_url)
    # Set submerged lands boundary polygon CRS to match AIS data crs
    if state_waters_gdf.crs != gdf.crs:
        state_waters_gdf = state_waters_gdf.to_crs(gdf.crs)

    # Spatial join (point-in-polygon)
    # Determine which AIS points are within the state submerged land polygon
    print("Tagging points inside state waters polygon...")
    gdf = gpd.sjoin(gdf, state_waters_gdf[['geometry']], how='left', predicate='within')
    gdf['IN_STATE'] = ~gdf['index_right'].isna()

    # Detect crossing from state to federal waters
    print("Detecting zone transitions (State vs Federal)...")
    gdf['ZONE'] = gdf['IN_STATE'].map({True: 'State', False: 'Federal'})
    gdf['PREV_ZONE'] = gdf.groupby('MMSI')['ZONE'].shift(1)

    # A crossing occurs when the zone changes
    gdf['ZONE_CROSSING'] = (gdf['ZONE'] != gdf['PREV_ZONE']) & gdf['PREV_ZONE'].notna()

    # GROUP BY CONTINUOUS STAY AND VALIDATE DURATION
    # Create a unique ID for every vessel for every continuous period in a single zone
    gdf['STAY_ID'] = gdf.groupby('MMSI')['ZONE'].transform(lambda x: (x != x.shift()).cumsum())

    # Calculate duration for every stay
    stay_durations = gdf.groupby(['MMSI', 'STAY_ID'])['BASEDATETIME'].agg(['min', 'max'])
    stay_durations['HOURS'] = (stay_durations['max'] - stay_durations['min']).dt.total_seconds() / 3600

    # Identify stays in state/federal waters that meet the 1-hour threshold
    # This avoids creating a new trip every time a vessel crosses the state waters boundary
    valid_map = stay_durations['HOURS'] >= 1.0
    gdf = gdf.join(valid_map.rename('IS_VALID_TRIP_ZONE'), on=['MMSI', 'STAY_ID'])
    
    # Line segment creation
    print("Creating movement segments...")
    # Get previous status to know when there are status changes
    gdf['PREV_STATUS'] = gdf.groupby('MMSI')['STATUS'].shift(1)
    # Status transition logic
    print("Processing status transitions (Anchored/Moored)...")
    parked_statuses = [1, 5]
    # Flag: Transition FROM 1 or 5 TO something else (Starting a trip)
    gdf['LEFT_PARKED'] = (
        (~gdf['STATUS'].isin(parked_statuses)) & 
        (gdf['PREV_STATUS'].isin(parked_statuses))
    )
    
    # TRIP SEGMENTATION USING RULES
    print("Segmenting trips based on all rules...")
    gdf['TIME_DIFF'] = gdf.groupby('MMSI')['BASEDATETIME'].diff().dt.total_seconds() / 3600
    
    # New trip if: 
    # Crossed into state waters and/or came back into federal waters for >1 OR 8hr gap OR Status change (In or Out of Parked) OR First Point
    gdf['TRIP_START'] = (
    (gdf['ZONE_CROSSING'] & gdf['IS_VALID_TRIP_ZONE']) | 
    (gdf['TIME_DIFF'] > 8) |
    (gdf['LEFT_PARKED'] == True)
    )

    # Ensure the very first point of every vessel starts a trip
    gdf.loc[gdf.groupby('MMSI').head(1).index, 'TRIP_START'] = True
    
    # Generate unique IDs
    gdf['TRIP_ID'] = gdf.groupby('MMSI')['TRIP_START'].cumsum()

    # CONVERT POINTS TO TRACKLINES
    print("Building tracklines with location filters...")
    gdf['POINTS_IN_TRIP'] = gdf.groupby(['MMSI', 'TRIP_ID'])['GEOMETRY'].transform('count')
    gdf = gdf[gdf['POINTS_IN_TRIP'] >= 2].copy()
    # Build tracklines
    lines_series = gdf.groupby(['MMSI', 'TRIP_ID'])['GEOMETRY'].apply(
        lambda x: LineString(x.tolist())) 

    # Get metrics including the zone tag
    # Use 'first' for the zone, assuming the trip mostly stays in the zone that triggered it
    metrics = gdf.groupby(['MMSI', 'TRIP_ID']).agg({
        'BASEDATETIME': ['min', 'max'],
        'ZONE': 'first',
        'STATUS': 'first'
    }).reset_index()
    metrics.columns = ['MMSI', 'TRIP_ID', 'START_TIME', 'END_TIME', 'TRIP_LOCATION', 'START_STATUS']

    final_lines = gpd.GeoDataFrame(metrics, geometry=lines_series.values, crs="EPSG:4326")
    
    # Nautical Miles Calculation (UTM 18N)
    final_lines_utm = final_lines.to_crs(epsg=32618)
    final_lines['DIST_NM'] = (final_lines_utm.geometry.length * 0.000539957)

    # EXPORT
    print(f"Saving tracklines to {output_layer}...")
    final_lines.to_file(
        str(line_gdb), 
        layer=output_layer, 
        driver="OpenFileGDB", 
        engine="pyogrio",
        layer_options={'TARGET_ARCGIS_VERSION': 'ARCGIS_PRO_3_2_OR_LATER'}
    )

    # EXPORT INDIVIDUAL FEATURE CLASSES FOR EACH MMSI
    print("Exporting individual MMSI layers...")
    unique_mmsis = final_lines['MMSI'].unique()

    for mmsi in unique_mmsis:
        # Filter for the specific vessel
        vessel_gdf = final_lines[final_lines['MMSI'] == mmsi]
        
        # Define a clean layer name
        vessel_layer_name = f"vessel_{int(mmsi)}_lines"
        
        vessel_gdf.to_file(
            str(line_gdb), 
            layer=vessel_layer_name, 
            driver="OpenFileGDB", 
            engine="pyogrio",
            layer_options={'TARGET_ARCGIS_VERSION': 'ARCGIS_PRO_3_2_OR_LATER'}
        )
    
    print(f"Finished exporting {len(unique_mmsis)} individual vessel layers.")
    
    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    run_trackline_pipeline()