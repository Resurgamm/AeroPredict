import requests
import pandas as pd
from datetime import datetime

AVIATIONSTACK_API_KEY = 'YOUR_AVIATIONSTACK_API_KEY'  # Replace with your Aviationstack key

AVIATION_BASE_URL = 'http://api.aviationstack.com/v1'
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Target Columns from your request
ALL_NUMERICAL_COLS = [
    'O_TEMP', 'O_PRCP', 'O_WSPD', 'D_TEMP', 'D_PRCP', 'D_WSPD', 
    'O_LATITUDE', 'O_LONGITUDE', 'D_LATITUDE', 'D_LONGITUDE',
    'DEP_DELAY'
]

ALL_CATEGORICAL_COLS = [
    'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'CRS_ARR_TIME'
]

class AviationDataExtractor:
    def __init__(self, aviation_key):
        self.aviation_key = aviation_key
        # Cache for airport coordinates to avoid repeated API calls
        self.airport_cache = {} 
        # Cache for weather to avoid hitting limits or repeated requests
        self.weather_cache = {}

    def _make_aviation_request(self, endpoint, params=None):
        """Helper to make Aviationstack API requests."""
        if params is None:
            params = {}
        params['access_key'] = self.aviation_key
        
        try:
            url = f"{AVIATION_BASE_URL}/{endpoint}"
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data:
                print(f"Aviation API Error: {data['error']}")
                return None
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Aviation Request failed: {e}")
            return None

    def get_airport_coordinates(self, iata_code):
        """
        Fetches Latitude and Longitude for a specific airport IATA code.
        """
        if iata_code in self.airport_cache:
            return self.airport_cache[iata_code]

        params = {'iata_code': iata_code}
        data = self._make_aviation_request('airports', params)

        if data and data.get('data'):
            airport = data['data'][0]
            try:
                lat = float(airport['latitude'])
                lon = float(airport['longitude'])
                self.airport_cache[iata_code] = (lat, lon)
                return lat, lon
            except (ValueError, TypeError):
                pass
        
        return None, None

    def get_weather_data(self, lat, lon):
        """
        Fetches current weather (Temp, Prcp, Wind) from Open-Meteo.
        Returns: (temp_celsius, prcp_mm, wind_speed_ms)
        """
        if lat is None or lon is None:
            return None, None, None
            
        # Create a cache key
        cache_key = (round(lat, 4), round(lon, 4))
        if cache_key in self.weather_cache:
            return self.weather_cache[cache_key]

        # Open-Meteo params
        # current parameters: temperature_2m, precipitation, wind_speed_10m
        # wind_speed_unit: ms (meters per second)
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'temperature_2m,precipitation,wind_speed_10m',
            'wind_speed_unit': 'ms' 
        }

        try:
            response = requests.get(OPEN_METEO_URL, params=params)
            if response.status_code == 200:
                data = response.json()
                
                # Check if 'current' data is available
                if 'current' in data:
                    current = data['current']
                    temp = current.get('temperature_2m')
                    prcp = current.get('precipitation', 0.0) # Default to 0 if not present
                    wspd = current.get('wind_speed_10m')
                    
                    result = (temp, prcp, wspd)
                    self.weather_cache[cache_key] = result
                    return result
            else:
                print(f"Weather API Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Weather Fetch Failed: {e}")

        return None, None, None

    def extract_flight_data(self, flight_limit=10):
        print(f"Fetching {flight_limit} flights...")
        
        params = {'limit': flight_limit}
        raw_data = self._make_aviation_request('flights', params)
        
        if not raw_data:
            return pd.DataFrame()

        processed_rows = []

        for flight in raw_data.get('data', []):
            try:
                # 1. Extract Categorical Data
                origin_iata = flight['departure']['iata']
                dest_iata = flight['arrival']['iata']
                crs_dep = flight['departure']['scheduled']
                crs_arr = flight['arrival']['scheduled']
                
                # Extract Departure Delay 
                # Strategy 1: Direct field from API
                dep_delay = flight['departure'].get('delay')
                
                # Strategy 2: Calculate from Actual vs Scheduled if direct field is None
                if dep_delay is None:
                    try:
                        sched_str = flight['departure'].get('scheduled')
                        actual_str = flight['departure'].get('actual')
                        
                        if sched_str and actual_str:
                            # Parse ISO format times (e.g. 2023-10-25T10:00:00+00:00)
                            # Ensure compatibility by replacing 'Z' if present
                            sched_str = sched_str.replace('Z', '+00:00')
                            actual_str = actual_str.replace('Z', '+00:00')
                            
                            t_sched = datetime.fromisoformat(sched_str)
                            t_actual = datetime.fromisoformat(actual_str)
                            
                            # Calculate difference in minutes
                            diff_seconds = (t_actual - t_sched).total_seconds()
                            dep_delay = diff_seconds / 60.0
                    except (ValueError, TypeError):
                        pass

                # Default to 0.0 if still None (assumed on time or data unavailable)
                if dep_delay is None:
                    dep_delay = 0.0
                else:
                    dep_delay = float(dep_delay)

                if not (origin_iata and dest_iata):
                    continue

                # 2. Extract Coordinates
                o_lat, o_lon = self.get_airport_coordinates(origin_iata)
                d_lat, d_lon = self.get_airport_coordinates(dest_iata)

                # 3. Extract Weather Data (Current Weather via Open-Meteo)
                o_temp, o_prcp, o_wspd = self.get_weather_data(o_lat, o_lon)
                d_temp, d_prcp, d_wspd = self.get_weather_data(d_lat, d_lon)

                row = {
                    # Categorical
                    'ORIGIN': origin_iata,
                    'DEST': dest_iata,
                    'CRS_DEP_TIME': crs_dep,
                    'CRS_ARR_TIME': crs_arr,
                    
                    # Numerical - Geo
                    'O_LATITUDE': o_lat,
                    'O_LONGITUDE': o_lon,
                    'D_LATITUDE': d_lat,
                    'D_LONGITUDE': d_lon,

                    # Numerical - Delay
                    'DEP_DELAY': dep_delay,

                    # Numerical - Weather (Populated)
                    'O_TEMP': o_temp, 'O_PRCP': o_prcp, 'O_WSPD': o_wspd,
                    'D_TEMP': d_temp, 'D_PRCP': d_prcp, 'D_WSPD': d_wspd
                }
                
                processed_rows.append(row)
                
            except (KeyError, TypeError) as e:
                continue

        # Create DataFrame
        df = pd.DataFrame(processed_rows)
        
        # Ensure all columns exist
        all_cols = ALL_CATEGORICAL_COLS + ALL_NUMERICAL_COLS
        for col in all_cols:
            if col not in df.columns:
                df[col] = None
                
        return df[all_cols]

if __name__ == "__main__":
    # IMPORTANT: Only Aviationstack key is needed now
    extractor = AviationDataExtractor(AVIATIONSTACK_API_KEY)
    
    # Extract data
    df_result = extractor.extract_flight_data(flight_limit=5)
    
    print("\n--- Extracted Data Sample ---")
    print(df_result.head())
    
    print("\n--- Data Info ---")
    print(df_result.info())
    
    # Save to CSV
    df_result.to_csv('raw_flights_data_10k.csv', index=False)