
# FT-100 Stock Screening and ML forecasting 

The project goal was to create a predictive model capable of identifying trading opportunities. The model operates on historical daily stock price data that has the data features ‘open’, ‘high’, ‘low’, ‘close’ and ‘volume’ (OHLCV) for stocks within the FT-100 index. This financial index is made up of the largest 100 public companies trading on the London Stock Exchange.

The machine learning (ML) software developed to achieve this goal is called ‘Ginger’. It runs as a FastAPI web application.

This is a final project submission for a Code Labs Academy Data Science and AI Bootcamp


![Logo](https://github.com/colinmoughton/Ginger/blob/master/static/images/logo.png)


## Documentation

[Documentation](https://github.com/colinmoughton/Ginger/blob/master/docs/Ginger_Report_Rev1.pdf)


## Authors

- [Colin Moughton](https://www.github.com/colinmoughton)


## Screenshots

![App Screenshot](https://github.com/colinmoughton/Ginger/blob/master/docs/Images/8-TrainingResAndBackTestStart.png)


## Installation

Install Ginger with Docker

```bash
  docker build -t ginger .
  docker run -p 8000:8000 ginger
```
    