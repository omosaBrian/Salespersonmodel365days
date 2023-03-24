from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
# read in CSV file
data = pd.read_csv('prdctslsvatlstngsumry.csv')

# convert the date column to datetime format
data['date'] = pd.to_datetime(data['date'])

# group the data by date and calculate the net sales for each day
daily_sales = data.groupby('date').agg({'net sales': 'sum'}).reset_index()

# set the date column as the index
daily_sales.set_index('date', inplace=True)

# resample the daily sales data to monthly frequency and calculate the total sales for each month
monthly_sales = daily_sales.resample('MS').sum()

# use monthly sales data to train the model
model = auto_arima(monthly_sales['net sales'], seasonal=False)

# predict net sales for the next 12 months from the last month of the dataset
last_month = monthly_sales.index[-1]
future_months = pd.date_range(start=last_month + pd.DateOffset(months=1), periods=12, freq='MS')
forecast = model.predict(n_periods=12)

# create a dataframe to store the forecasts
forecast_df = pd.DataFrame({'date': future_months, 'total sales Forecast': forecast})
forecast_df.set_index('date', inplace=True)
# print out the forecasted net sales for the next 12 months
print("Forecasted net sales for the Next 12 Months:\n", forecast_df)
# evaluate the performance of the model using advanced metrics
preds = model.predict(n_periods=len(monthly_sales))
mae = mean_absolute_error(monthly_sales['net sales'], preds)
mse = mean_squared_error(monthly_sales['net sales'], preds)

# print out the performance metrics of the model
print("\nPerformance Metrics:")
print("MAE: ", mae)
print("MSE: ", mse)
# save the trained ARIMA model as a joblib file
joblib.dump(model, 'model.joblib')

# Load the saved model
model = joblib.load('model.joblib')

# Initialize the Flask application
app = Flask(__name__)

# Define an API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data
    data = request.get_json()

    # Convert the request data into a DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # Make a prediction
    forecast = model.predict(input_data)

    # Create a response dictionary
    response = {
        'prediction': forecast.tolist()
    }

    # Return the response as JSON
    return jsonify(response)
