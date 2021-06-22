import pandas as pd

def add_weatherFeature(region: str, feature: str, weather_history: pd.core.frame.DataFrame):
    '''
    This function returns the feature [tempC, tempF,winddirDegree...]
    of weather choosen by region with resample period.
    '''
    
    feature_index = weather_history[weather_history['region'] == region].index
    return(weather_history[weather_history['region'] == region].groupby(by=feature_index).mean().resample('1H').bfill()[feature])

def preprocess_data(ercot_hourly: pd.core.frame.DataFrame, weather_history: pd.core.frame.DataFrame):
    
    ercot_hourly.fillna(method='backfill',inplace=True)
    # Convert CST & CDT to UTC
    ercot_hourly['Hour_Ending']= pd.to_datetime(ercot_hourly['Hour_Ending'],utc= True)
    # Datetimeindex
    ercot_hourly.set_index("Hour_Ending", inplace=True)
    ercot_hourly.index = ercot_hourly.tz_localize(tz=None, copy=True).index

    # transform time into hours:minutes
    time = {'time':{0:'00:00',300:'03:00',600:'06:00',900:'09:00',1200:'12:00',
                    1500:'15:00',1800:'18:00',2100:'21:00'}}
    weather_history.replace(time, inplace=True)

    # join date + time
    weather_history['date'] = weather_history.date + " " + weather_history.time.map(str)
    weather_history.drop(columns={'time'}, inplace=True)

    # make timeindex
    weather_history.date = pd.DatetimeIndex(weather_history.date)
    weather_history.set_index('date', inplace=True)

    # transform city to region
    city_region = {'Wichita Falls':'North',
                   'Tyler':'East',
                   'Corpus Christi':'South',
                   'Brownsville':'South',
                   'Dallas':'North Central',
                   'Austin':'South Central',
                   'Midland':'Far West',
                   'San Antonio':'South Central',
                   'Houston':'Coast',
                   'Abilene':'West'}

    weather_history['region'] = weather_history.city.replace(city_region) 

