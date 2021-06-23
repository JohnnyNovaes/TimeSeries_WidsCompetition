<meta name="author" content="[João Henrique Saraceni Novaes]">
<p align="center">
  <img src="https://www.thedatalab.com/wp-content/uploads/2018/01/WiDS-datathon.png" />
</p>

# Background
Load forecasting is the prediction of electrical power required to meet the short or long-term demand. The forecasting helps utility companies plan on their capacity to keep the electricity running in every household and business.

The Electric Reliability Council of Texas (ERCOT) supplies power to more than 25 million Texas customers and represents 90 percent of the state's electric load. As weather influences the electrical demand significantly, ERCOT divides its service region into 8 weather zones within each of which weather is usually similar. The electrical load is reported on an hourly basis in each weather zone.

# Overview
This year's challenge will focus on forecasting hourly electrical load in each weather zone in the short-term (the next 7 days). Participants will build models to learn how the electrical load was inference by key factors (e.g., weather) using historical data and make the forecast to a near-future period.

After the competition starts on May 17, participants have about 4 weeks to work on it. By June 12 23:59 CDT, participants are expected to submit their load forecast between June 13 00:00 CDT and June 20 00:00 CDT.

While key data is publicly available, we created a data repository for participants' convenience (the repository will be available to the public after the competition starts). In the repository, the following data is updated on a daily basis:

Hourly electrical load in each ECROT weather zone (since 2005)
Tri-hourly weather in major cities cross ECROT weather zones (since 2008)
Tri-hourly weather forecast in major cities cross ECROT weather zones (next 2 weeks)
Daily COVID data in each Texas county (since early 2020)
Participants are encouraged to use additional publicly available data besides the data repository if they believe it could improve the model.

![alt text](https://github.com/JohnnyNovaes/TimeSeries_WidsCompetition/blob/main/data/ercotWeatherZoneMap.png?raw=true)
