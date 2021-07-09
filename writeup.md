## Developing a Neural Network to Forecast U.S. Treasury Interest Rates
I developed a recurrent neural network (RNN) to forecast U.S. Treasury interest rates, most notably the 10-year bond rate as this value is used most frequently as a benchmark for a risk-free rate. 

After evaluating several tests with different variables (batch size, epochs, dropout rate, LSTM neurons, and time duration for the dataset) it was found that forecasting could be achieved with a relatively low root mean squared error (RMSE) for either the shortest duration bill rates (one-month through one-year) or the longest (seven-year through thirty-year) but a cohesive forecasting network could not be achieved for all eleven interest rates evaluated. 


### Design
After reading some literature on the topic and finding evidence that a neural net model can be used to predict U.S. Treasury interest rates, I evaluated what variables should be included and to what extent. In light of a Federal Reserve Bank of San Francisco Economic letter, dated June 27, 2016, "some of the published evidence on the predictive power of macroeconomic variables may be spurious, supporting the more traditional view that current interest rates contain all the relevant information for predicting future interest rates" ([link](https://www.frbsf.org/economic-research/files/el2016-20.pdf)). This cemented my decision to forecast future rates based purely on the pre-existing interest rate history, rather than include economic variables. 

### Data
I collected the Daily Treasury Yield Curve Rates , accessible through the U.S. Department of the Treasury's website ([link](https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield)), with data going back to January 1, 2000. I also collected the Daily Treasury Real Yield Curve Rates ([link](https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=realyield)) with data going back to January 2, 2003. For both sets of data longer time frames were tested (with max size of 59,191 data points) but a time frame of January 1, 2015 to July 2, 2021 (17,908 data points) was ultimately used to only collect signals from relatively recent interest rate adjustments. 

Missing days like weekends and holidays were treated as extensions of the preceding day, with interest rates copied over. Averaging the rates that came before and after missing days was considered, but discarded because that would imply that future rates are known prior to their release. Testing the neural net with weekend days excluded did not improve the testing outcomes with any variable changes.

### Algorithms
After testing both the reLu and Swish activation functions, I opted to use Swish as it resulted in slightly more accurate outcomes across most iterations. The Swish activation function was introduced by Google in 2017 and can be modeled as f(x) = x * sigmoid(x). 

I used multiple Keras tools including LSTM with 50 neurons, as well as a Dropout layer with a probability of 20%, and compiled using Mean Abolute Error (MAE) and Adam optimization, which is a stochastic gradient descent method. 


### Tools
Pandas: for data cleaning and manipulation (preprocessing.py)

Keras: for neural net production and testing (lstm_final_multivariate.py)

Sklearn: for mean squared error calculation and MinMaxScaler



### Communication
I developed a PowerPoint to share my findings with several visuals of the data, shown below. 

![image](https://user-images.githubusercontent.com/71529189/125053079-a986a800-e072-11eb-876e-a7893e8727c7.png)

![image](https://user-images.githubusercontent.com/71529189/125053097-b1464c80-e072-11eb-8ddf-07fcc3a408aa.png)

