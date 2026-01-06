
import pandas as pd
import numpy as np

# ----------------------------------------------- #
#           DATA PROCESSING FUNCTIONS             #


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the earth (specified in decimal degrees).
    """
    R = 6371  # Earth radius in kilometers

    # Convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def load_demand_and_airport_data(file_path="DemandGroup6.xlsx"):
    """
    Loads airport data and the demand matrix from the specified Excel file.
    This function reads the whole sheet and slices the required data blocks.
    It handles non-numeric values in the demand block by coercing them to numbers.
    The source file is missing one column of demand data, so it's padded with zeros.

    Args:
        file_path (str): The path to the DemandGroup6.xlsx file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Airport metadata (ICAO, Latitude, Longitude, Runway).
            - pd.DataFrame: The 20x20 demand matrix with ICAO codes as index and columns.
    """
    # Read the entire sheet at once to avoid complex parsing arguments
    full_sheet = pd.read_excel(file_path, header=None)

    # Extract airport data block (Cells C3:V7)
    airport_block = full_sheet.iloc[3:8, 2:22].copy()
    airport_df = airport_block.T
    airport_df.columns = ['AirportName', 'ICAO', 'Latitude (deg)', 'Longitude (deg)', 'Runway length (m)']
    airport_df.set_index('ICAO', inplace=True)
    # Coerce coordinate columns to numeric, as they might be read as objects
    for col in ['Latitude (deg)', 'Longitude (deg)']:
        airport_df[col] = pd.to_numeric(airport_df[col], errors='coerce')


    # Extract demand data block (Cells D8:V27)
    demand_block = full_sheet.iloc[11:30, 1:22]                        
    demand_block.columns = full_sheet.iloc[10, 1:22] 

    # Create the final 20x20 demand dataframe
    demand_block.set_index(demand_block.columns[0], inplace=True)      
    demand_df = demand_block 

    return airport_df, demand_df

def calculate_distance_matrix(airports_df):
    """
    Calculates the distance matrix between all airports using the Haversine formula.

    Args:
        airports_df (pd.DataFrame): DataFrame with airport data, including latitude and longitude.

    Returns:
        pd.DataFrame: A matrix with the great-circle distance in km between each pair of airports.
    """
    icao_codes = airports_df.index
    dist_matrix = pd.DataFrame(index=icao_codes, columns=icao_codes, dtype=float)

    for origin_icao, origin_data in airports_df.iterrows():
        for dest_icao, dest_data in airports_df.iterrows():
            if origin_icao == dest_icao:
                dist_matrix.loc[origin_icao, dest_icao] = 0
            else:
                lat1 = origin_data['Latitude (deg)']
                lon1 = origin_data['Longitude (deg)']
                lat2 = dest_data['Latitude (deg)']
                lon2 = dest_data['Longitude (deg)']
                dist = haversine(lat1, lon1, lat2, lon2)
                dist_matrix.loc[origin_icao, dest_icao] = dist
    
    dist_matrix.index.name = "ICAO_From"
    dist_matrix.columns.name = "ICAO_To"
    return dist_matrix

def load_fleet_data(file_path="FleetType.xlsx"):
    """
    Loads the fleet data from the specified Excel file, transposes it,
    and renames the columns to be more Python-friendly.

    Args:
        file_path (str): The path to the FleetType.xlsx file.

    Returns:
        pd.DataFrame: A DataFrame with fleet specifications, indexed by aircraft type.
    """
    fleet_df = pd.read_excel(file_path, index_col=0).T
    fleet_df.index.name = "Aircraft Type"

    # Define the mapping from old column names to new, Python-friendly names
    column_mapping = {
        'Speed [km/h]': 'speed_kmh',
        'Seats': 'seats',
        'Average TAT [min]': 'tat_min',
        'Maximum Range [km]': 'range_km',
        'Runway Required [m]': 'runway_m',
        'Lease Cost [€/day]': 'lease_cost_eur_day',
        'Fixed Operating Cost (Per Fligth Leg)  [€]': 'fixed_op_cost_eur_flight',
        'Cost per Hour': 'time_based_cost_eur_hour',
        'Fuel Cost Parameter': 'fuel_cost_eur_kg',
        'Fleet': 'fleet_size'
    }
    
    # Rename the columns
    fleet_df.rename(columns=column_mapping, inplace=True)
    
    return fleet_df


def load_hour_coefficients(file_path="HourCoefficients.xlsx"):
    """
    Loads the hourly demand coefficients from the specified Excel file.

    Args:
        file_path (str): The path to the HourCoefficients.xlsx file.

    Returns:
        pd.DataFrame: A DataFrame with hourly coefficients, indexed by ICAO code.
    """
    headers = list(range(24))
    hour_coeffs_df = pd.read_excel(
        file_path,
        skiprows=1,
        header=None,
        names=['Hour_of_Day', 'Airport', 'ICAO'] + headers
    )
    hour_coeffs_df = hour_coeffs_df.set_index('ICAO').drop(columns=['Hour_of_Day', 'Airport'])
    # The first row is junk, let's drop rows where the index is NaN
    hour_coeffs_df = hour_coeffs_df[hour_coeffs_df.index.notna()]
    return hour_coeffs_df


def main():
    """
    Main function to demonstrate the data loading functions.
    """
    print("--- Loading Airport and Demand Data ---")
    airports, demand = load_demand_and_airport_data()
    print("\nAirport Data:")
    print(airports.head())
    print("\nDemand Matrix:")
    print(demand.head())

    print("\n--- Calculating Distance Matrix ---")
    distance_matrix = calculate_distance_matrix(airports)
    print(distance_matrix.head())

    print("\n--- Loading Fleet Data ---")
    fleet = load_fleet_data()
    print(fleet.head())

    print("\n--- Loading Hour Coefficients ---")
    hour_coeffs = load_hour_coefficients()
    print(hour_coeffs.head())


if __name__ == "__main__":
    main()
