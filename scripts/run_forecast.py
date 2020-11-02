# -*- encoding: utf-8 -*-
'''
@Filename    : run_forecast.py
@Datetime    : 2020/09/27 18:35:44
@Author      : Joe-Bu
@version     : 1.0
'''

import sys
sys.path.append("../")

from utils.ConfigParseUtils import ConfigParser

from src.DataForecast.run import forecast_arima
from src.DataForecast.run import forecast_fbprophet
from src.DataForecast.run import forecast_lstm
from src.DataForecast.run import forecast_gru


def main():
    """
    Forecast Workflow.
    """
    cfp = ConfigParser()
    if cfp.start is not None and cfp.end is not None:
        assert cfp.start <= cfp.end
    
    if cfp.model == "Arima":
        forecast_arima(station=cfp.name, index=cfp.index, start=cfp.start, end=cfp.end)

    elif cfp.model == "Fbprophet":
        forecast_fbprophet(station=cfp.name, index=cfp.index, start=cfp.start, end=cfp.end)

    elif cfp.model == "LSTM":
        forecast_lstm(station=cfp.name, index=cfp.index, start=cfp.start, end=cfp.end)

    elif cfp.model == "GRU":
        forecast_gru(station=cfp.name, index=cfp.index, start=cfp.start, end=cfp.end)

    else:
        print("Wrong model with {}".format(cfp.model))
        sys.exit()


if __name__ == "__main__":
    main()
