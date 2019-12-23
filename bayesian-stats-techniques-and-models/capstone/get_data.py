import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# https://maps.waterdata.usgs.gov/mapper/
# https://maps.waterdata.usgs.gov/mapper/
# https://waterdata.usgs.gov/nwis/current/?type=precip&group_key=state_cd
# https://waterservices.usgs.gov/rest/DV-Test-Tool.html
# http://cdec.water.ca.gov/jspplot/jspPlotServlet.jsp?sensor_no=11108&end=12%2F14%2F2019+22%3A25&geom=huge&interval=600&cookies=cdec01
# http://cdec.water.ca.gov/dynamicapp/staMeta?station_id=TK2
# http://cdec.water.ca.gov/dynamicapp/selectSnow
# https://www.nohrsc.noaa.gov/interactive/html/graph.html?station=TADC1&w=600&h=400&o=a&uc=0&by=2015&bm=12&bd=10&bh=6&ey=2019&em=12&ed=12&eh=5&data=2&units=0&region=us
# http://cdec.water.ca.gov/dynamicapp/wsSensorData
# http://cdec.water.ca.gov/dynamicapp/QueryDaily?s=TK2&end=2019-12-15&span=100weeks
# http://cdec.water.ca.gov/dynamicapp/querySWC?reg=NORTH
# Donner "http://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet?Stations=DNS&SensorNums=18&dur_code=M&Start=2017-01&End="
# https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
# https://www.ncdc.noaa.gov/data-access

# https://www.ncdc.noaa.gov/cdo-web/datasets/NORMAL_DLY/locations/ZIP:96161/detail#stationlist
# https://www.ncdc.noaa.gov/cdo-web/datasets

# Truckee at reno: 10348000


# Rivers
# Lake Tahoe: 391359120012701
# Donner: 10338400
# Boca: 10344490

START = '2017-01-01'
END = '2019-12-01'

RIVER_URL = "https://waterservices.usgs.gov/nwis/dv/?format=json&sites={site_id}&startDT={START}&endDT={END}&parameterCd=00060"
WEATHER_URL = f'https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USS0020K13S&startDate={START}&endDate={END}&format=JSON&units=metric'
TRUCKEE = 10346000
TAHOE = 10337500
DONNER = 10338500
BOCA = 10344500


def get_river_data(url):
    res = json.loads(requests.get(url).content)

    datatime_str = "%Y-%m-%dT%H:%M:%S.%f"
    d = res['value']['timeSeries'][0]['values'][0]['value']

    data = list(zip(*[
        [datetime.strptime(datum['dateTime'], datatime_str), float(datum['value'])]
        for datum in d
    ]))

    data_dict = {'time': data[0], 'value': data[1]}

    df = pd.DataFrame(data=data_dict)

    def _clean_river_data(df):
        drop_idx = df.value[df.value == 0].index
        df.drop(drop_idx, inplace=True)
        df.value = df.value.apply(lambda x: .02832*x)  # ft^3 to m^3
        return df

    return _clean_river_data(df)


def get_climate_data():
    res = json.loads(requests.get(WEATHER_URL).content)
    df = pd.DataFrame(res)
    df = df.drop(columns=['STATION', 'TMIN', 'TMAX', 'TOBS'])

    cols = df.columns.drop('DATE')
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    datatime_str = "%Y-%m-%d"
    df.DATE = df.DATE.apply(
        lambda x: datetime.strptime(x, datatime_str)
    )
    return df


def _get_from_file(data_name):
    return pd.read_pickle(f'data/{data_name}.pkl')


def get_river_fn(df):
    return interp1d(df.time.values.astype(float), df.value.values, bounds_error=False, kind='linear')


def get_weather_fn(df):
    def _get_fn(time, values):
        return interp1d(time.astype(float), values, bounds_error=False, kind='linear')

    cols = df.columns.drop('DATE')
    funs = {
        col: _get_fn(df.DATE.values, df[col].values) for col in cols
    }
    return funs


truckee_data = get_river_data(RIVER_URL.format(site_id=TRUCKEE, START=START, END=END))
tahoe_data = get_river_data(RIVER_URL.format(site_id=TAHOE, START=START, END=END))
donner_data = get_river_data(RIVER_URL.format(site_id=DONNER, START=START, END=END))
boca_data = get_river_data(RIVER_URL.format(site_id=BOCA, START=START, END=END))

weather_data = get_climate_data()
weather_fns = get_weather_fn(weather_data)

truckee_fn = get_river_fn(truckee_data)
tahoe_fn = get_river_fn(tahoe_data)
boca_fn = get_river_fn(boca_data)
donner_fn = get_river_fn(donner_data)

_time = truckee_data.time.values
time = np.arange(_time[0], _time[-1], timedelta(days=1), dtype=_time.dtype)
float_time = time.astype(int)

river_data = pd.DataFrame({
    'time': time,
    'truckee': truckee_fn(float_time),
    'tahoe': tahoe_fn(float_time),
    'donner': donner_fn(float_time),
    'boca': boca_fn(float_time),
    'rain': weather_fns['PRCP'](float_time),
    'snow': weather_fns['WESD'](float_time),
    'temp': weather_fns['TAVG'](float_time),
})

plt.plot(river_data.truckee, river_data.tahoe + river_data.donner + river_data.boca, 'o')

river_data.to_pickle('./data/clean_data.pkl')
