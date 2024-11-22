num_columns = [
    'Daily Rainfall Total (mm)',
    'Highest 30 Min Rainfall (mm)',
    'Highest 60 Min Rainfall (mm)',
    'Highest 120 Min Rainfall (mm)',
    'Min Temperature (deg C)',
    'Maximum Temperature (deg C)',
    'Min Wind Speed (km/h)',
    'Max Wind Speed (km/h)',
    'Sunshine Duration (hrs)',
    'Cloud Cover (%)',
    'Wet Bulb Temperature (deg F)',
    'Relative Humidity (%)',
    'Air Pressure (hPa)',
    'pm25_north',
    'pm25_south',
    'pm25_east',
    'pm25_west',
    'pm25_central',
    'psi_north',
    'psi_south',
    'psi_east',
    'psi_west',
    'psi_central'
]

cat_columns = [
    'Dew Point Category',
    'Wind Direction',
    'Daily Solar Panel Efficiency'
]

Dew_Point_Category_replacements = {
    'VH': 'VERY HIGH',
    'H': 'HIGH',
    'M': 'MODERATE',
    'L': 'LOW',
    'VL': 'VERY LOW',
    'HIGH LEVEL': 'HIGH',
    'MINIMAL': 'VERY LOW',
    'BELOW AVERAGE': 'LOW',
    'NORMAL': 'MODERATE',
    'EXTREME': 'VERY HIGH'
}

Wind_Direction_replacements = {
    # Those with a "." at the end will be replaced in the code via str.replace
    'EAST': 'E',
    'NORTHEAST': 'NE',
    'SOUTHWARD': 'S',
    'SOUTHEAST': 'S',
    'NORTHWARD': 'N',
    'NORTHWEST': 'NW',
    'WEST': 'W',
    'NORTH': 'N',
    'SOUTH': 'S'
}

Dew_Point_Category_encoding = {'VERY HIGH': 5,
                               'HIGH': 4, 'MODERATE': 3, 'LOW': 2, 'VERY LOW': 1}

Daily_Solar_Panel_Efficiency_encoding = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}

DSPE_in_numbers = [
    v for v in Daily_Solar_Panel_Efficiency_encoding.values()]

DSPE_in_words = [
    k for k in Daily_Solar_Panel_Efficiency_encoding.keys()]

Daily_Solar_Panel_Efficiency_order = ['LOW', 'MEDIUM', 'HIGH']

pm25_and_psi_cols = [
    'pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 'pm25_central',
    'psi_north', 'psi_south', 'psi_east', 'psi_west', 'psi_central'
]

rainfall_cols = [
    'Daily Rainfall Total (mm)',
    'Highest 30 Min Rainfall (mm)',
    'Highest 60 Min Rainfall (mm)',
    'Highest 120 Min Rainfall (mm)'
]

# Color for console messages
W = "\033[0m"  # white (default)
R = "\033[31m"  # red
G = "\033[32m"  # green
B = "\033[34m"  # blue
O = "\033[33m"  # orange
P = "\033[35m"  # purple
