import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


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


RAIN_URL = "https://waterservices.usgs.gov/nwis/dv/?format=json&sites=391930120165301&period=P100W&siteStatus=all"
RIVER_URL = "https://waterservices.usgs.gov/nwis/dv/?format=json&sites={site_id}&period=P100W&parameterCd=00060"
LAKE_URL = "https://waterservices.usgs.gov/nwis/dv/?format=json&sites={site_id}&period=P100W&parameterCd=00065"
SNOW_URL = "https://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet?Stations=TK2&SensorNums=82&end=2019-12-15&span=100weeks"
TEMP_URL = "https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USS0020K13S&startDate=2016-01-01&endDate=2019-12-10&format=JSON"

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
    return df


def _get_from_url(data_name):
    url = {
        'river': RIVER_URL,
        'rain': RAIN_URL,
        'lake': LAKE_URL,
        'snow': SNOW_URL,
    }[data_name]

    res = json.loads(requests.get(url).content)

    if data_name == "snow":
        datatime_str = "%Y-%m-%d %H:%M"
        data = list(zip(*[
            [datetime.strptime(datum['obsDate'], datatime_str), float(datum['value'])]
            for datum in res
        ]))
    else:
        datatime_str = "%Y-%m-%dT%H:%M:%S.%f"
        d = res['value']['timeSeries'][0]['values'][0]['value']

        data = list(zip(*[
            [datetime.strptime(datum['dateTime'], datatime_str), float(datum['value'])]
            for datum in d
        ]))

    data_dict = {'time': data[0], 'value': data[1]}

    df = pd.DataFrame(data=data_dict)
    # df.set_index('time')
    df.to_pickle(f'data/{data_name}.pkl')
    return df


def _clean_snow_data(df):
    df.value = df.value.apply(lambda x: max(0, x))
    return df


def _clean_river_data(df):
    drop_idx = df.value[df.value == 0].index
    df.drop(drop_idx, inplace=True)
    return df


def _get_from_file(data_name):
    return pd.read_pickle(f'data/{data_name}.pkl')


def get_data(data_name):
    # return _get_from_url(data_name)
    return _get_from_file(data_name)


def get_climate_data():
    url = 'https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USS0020K13S&startDate=2017-01-01&endDate=2019-12-10&format=JSON&units=imperial'
    res = json.loads(requests.get(url).content)
    df = pd.DataFrame(res)
    cols = df.columns.drop('DATE', 'STATION')
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df


def get_value_fn(df):
    return interp1d(df.time.values.astype(float), df.value.values, bounds_error=False, kind='linear')


def moving_average(a, n=5):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[n - 1:] /= n
    return ret


truckee_data = get_river_data(RIVER_URL.format(site_id=TRUCKEE))
truckee_data = _clean_river_data(truckee_data)
tahoe_data = get_river_data(RIVER_URL.format(site_id=TAHOE))
donner_data = get_river_data(RIVER_URL.format(site_id=DONNER))
boca_data = get_river_data(RIVER_URL.format(site_id=BOCA))
# tahoe_data = _clean_river_data(truckee_data)

truckee_data.plot('time', 'value')
tahoe_data.plot('time', 'value')
donner_data.plot('time', 'value')
boca_data.plot('time', 'value')

rain_data = get_data('rain')
snow_data = _clean_snow_data(get_data('snow'))

truckee_fn = get_value_fn(truckee_data)
rain_fn = get_value_fn(rain_data)
lake_fn = get_value_fn(tahoe_data)
snow_fn = get_value_fn(snow_data)
_time = truckee_data.time.values

time = np.arange(_time[0], _time[-1], timedelta(days=1), dtype=_time.dtype).astype(float)


fig, axs = plt.subplots(4, 1, sharex=True)

axs[0].plot(time, truckee_fn(time))
avg_rain = moving_average(rain_fn(time))
axs[1].plot(time, avg_rain)
axs[2].plot(time, lake_fn(time))
axs[3].plot(time, snow_fn(time))
