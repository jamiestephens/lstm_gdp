# Developing a LSTM model using daily yield curve data to predict monthly GDP rates one year out

## Metis Project VI: Deep Learning


### Design

Multiple sources point to U.S. Treasury interest rate spreads being used as a reliable leading indicator for changes in GDP roughly one year out. 

### Data

Daily yield curve data was derived from raw values released every weekday (excepting national holidays) from the Federal Reserve. Following Damodaran's work, I used the same five spreads: 1 year versus 3 months, 2 years versus 3 months, 5 years versus 2 years, 10 years versus 2 years, and 10 years versus 3 months.

Monthly Real GDP values (using the Brave-Butters-Kelley Real Gross Domestic Product) were collected from the Federal Reserve Bank of St. Louis (https://fred.stlouisfed.org/series/BBKMGDP).

### Process

I used a many-to-one sequenced LSTM, where each month's worth of data was aligned to the next reported GDP value, offset by one year. This meant, for example, that the model was fed November 2010 values in order to predict the BBK Real GDP value reported on December 1, 2011. Each spread value (five for each working day) is considered a feature. 


### Sources

Damodaran, A. (2018, December 7). <i>Is there a signal in the noise? Yield Curves, Economic Growth and Stock Prices!</i>  http://aswathdamodaran.blogspot.com/2018/12/is-there-signal-in-noise-yield-curves.html. 

<i>Yield Curve and Predicted GDP Growth.</i> Federal Reserve Bank of Cleveland. (2021, July 29). https://www.clevelandfed.org/our-research/indicators-and-data/yield-curve-and-gdp-growth.aspx. 