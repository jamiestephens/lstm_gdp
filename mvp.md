## Minimum Viable Product

I built a baseline univariate LSTM model that evaluated, from 01/01/2010 to 07/02/2021, the 10-year Treasury rate. This was done using Keras and TimeSeriesGenerator, with a relu activator function.  

The preprocessing for the data, which was sourced from the U.S. Treasury website, was done within preprocessing.py. The baseline model was completed within univariate_lstm_tsg.py.


![image](https://user-images.githubusercontent.com/71529189/124717394-1400e200-ded3-11eb-9917-80f70909d444.png)


For further evaluation, given the changing relationships in bond rates shown below, a multivariate approach will be taken using the same Keras tools to create better predictions for the 10 year bond rate.

![image](https://user-images.githubusercontent.com/71529189/124717239-efa50580-ded2-11eb-8061-76a0df52c32c.png)
