import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

LAKE_URL = "https://waterservices.usgs.gov/nwis/dv/?format=json&sites={site_id}&period=P100W&siteType=LK&siteStatus=all"
RIVER_URL = "https://waterservices.usgs.gov/nwis/dv/?format=json&sites={site_id}&period=P100W&siteType=ST&siteStatus=all"

tahoe = 10337500
donner = 10338400
boca = 10344490


def get_lake_data(url):
    res = json.loads(requests.get(url).content)

    datatime_str = "%Y-%m-%dT%H:%M:%S.%f"
    d = res['value']['timeSeries'][1]['values'][0]['value']

    data = list(zip(*[
        [datetime.strptime(datum['dateTime'], datatime_str), float(datum['value'])]
        for datum in d
    ]))

    data_dict = {'time': data[0], 'value': data[1]}

    df = pd.DataFrame(data=data_dict)
    return df


tahoe_df = get_lake_data(RIVER_URL.format(site_id=tahoe))
donner_df = get_lake_data(LAKE_URL.format(site_id=donner))
boca_df = get_lake_data(LAKE_URL.format(site_id=boca))
