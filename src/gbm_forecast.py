from lightgbm import LGBMRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


class gbm_forecast():
    
    '''
    data: Ercot Hourly power consumption history (Kwm)
    region: Region of power consumption - [Coast, West, North, East, Far West, South, South Central]
    weather_history: Weather features history for each region [tempC, windspeed ...]
    weather_forecast: Next 2 weeks forecast of the weather features for each region

    A LightGBM model forecast the next 168 values , for a weekly power demand forecast.
    
    LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient.

    '''
    
    def __init__(self, data, region, weather_history, weather_forecast):
        self.region = region
        self.weather = weather_history.append(weather_forecast).copy()
        self.region_data = data[[region,'Hour_Ending']].copy()
        self.weather_forecast = weather_forecast.copy()

        # preprocess weather and region data
        self.preprocess_data()
        self.build_features()
        
        # train and forecast
        self.train()
        self.forecast()
        
    def add_weatherFeature(self, feature):
        feature_index = self.weather[self.weather['region'] == self.region].index
        return(self.weather[self.weather['region'] == self.region].groupby(by=feature_index).mean().resample('1H').bfill()[feature])
    
    def build_features(self):
        
        self.region_data['hour'] = self.region_data.index.hour
        self.region_data['year'] = self.region_data.index.year
        self.region_data['dayofyear'] = self.region_data.index.dayofyear
        self.region_data['DewPointC'] = self.add_weatherFeature('DewPointC')
        self.region_data['uvIndex'] = self.add_weatherFeature('uvIndex')
        self.region_data['last_hour'] = self.region_data[self.region].shift()
        self.region_data['last_hour_diff'] = self.region_data[self.region].shift().diff()
        self.region_data['last_2_hours_diff'] = self.region_data[self.region].shift(2).diff()
        self.region_data['roll_mean_3'] = self.region_data[self.region].shift(24).rolling(2).std()

  self.region_data['roll_mean_3'] = self.region_data[self.region].shift(24).rolling(2).std()
        
        self.region_data.dropna(inplace=True)
        
    def preprocess_data(self, data=None):
        # PREPROCESS WEATHER
        
        # transform time into hours:minutes
        time = {'time':{0:'00:00',300:'03:00',600:'06:00',900:'09:00',1200:'12:00',
                        1500:'15:00',1800:'18:00',2100:'21:00'}}
        self.weather.replace(time, inplace=True)

        # join date + time
        self.weather['date'] = self.weather.date + " " + self.weather.time.map(str)
        self.weather.drop(columns={'time'}, inplace=True)

        # make timeindex
        self.weather.date = pd.DatetimeIndex(self.weather.date)
        self.weather.set_index('date', inplace=True)

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

        self.weather['region'] = self.weather.city.replace(city_region)

        
        # PREPROCESS POWER DEMAND ERCOT HOURLY      
        
        # Convert CST & CDT to UTC
        self.region_data['Hour_Ending']= pd.to_datetime(self.region_data['Hour_Ending'],utc= True)
        # Datetimeindex
        self.region_data.set_index("Hour_Ending", inplace=True)
        self.region_data = self.region_data.tz_convert(tz='US/Central').tz_localize(tz=None)       
        
        # add hollidays
        dr = pd.date_range(start='2008-01-01', end='2021-06-20')
        cal = calendar()
        self.holidays = cal.holidays(start=dr.min(), end=dr.max())

        self.region_data['Holiday'] = self.region_data.index.isin(self.holidays)
        self.region_data['Holiday'] = self.region_data['Holiday'].astype(int)
        
    def train(self):
        
        train = self.region_data.reset_index(drop=True)
        
        params = {
            'boosting_type':'gbdt',
            'num_leaves':18,
            'learning_rate':0.08,
            'n_estimators':1500,
            'force_col_wise':True,
            'random_state':42,
            'silent':True
            }
        
        self.model = LGBMRegressor(**params)
        self.model.fit(train.drop([self.region],axis=1), np.log1p(train[self.region]))
        self.train_data = train
    
    def forecast(self):
        
        forecasted = []
        # week forecast
        for idx in range(24*7):
            region_data = self.region_data.iloc[[-1]].copy()

            point_value = region_data.drop([self.region],axis=1)

            forecasted.append(np.expm1(self.model.predict(point_value)[0]))

            if region_data.index.hour == 0:
                region_data['hour'] = 1
                region_data['dayofyear'] = region_data.index.dayofyear +1

            else:
                region_data['hour'] = region_data.index.hour + 1

            # make next dataframe
            region_data['last_hour'] = self.region_data[self.region].iloc[-1]
            region_data['last_hour_diff'] = self.region_data[self.region].diff().iloc[-1]
            region_data['last_2_hours_diff'] = self.region_data[self.region].shift(1).diff().iloc[-1]
            region_data['roll_mean_3'] = self.region_data[self.region].shift(23).rolling(2).std()[-1]
            region_data.index += pd.TimedeltaIndex([1], unit='h')
            region_data['DewPointC'] = self.add_weatherFeature('DewPointC')
            region_data['uvIndex'] = self.add_weatherFeature('uvIndex')
            region_data['Holiday'] = region_data.index.isin(self.holidays)
            region_data['Holiday'] = region_data['Holiday'].astype(int) 
            region_data[self.region] = np.expm1(self.model.predict(region_data.drop([self.region],axis=1)))

            # append new data to train 
            self.region_data = self.region_data.append(region_data)