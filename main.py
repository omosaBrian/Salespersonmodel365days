import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import shap
import plotly.express as px
from jupyter_dash import JupyterDash
import itertools

# Read in the Excel file
xls = pd.ExcelFile('prdctslsvatlstngsumry.xls')

# Read data from first sheet, parse the 'Date' column as dates and set it as the index
df = pd.read_excel(xls, sheet_name=xls.sheet_names[0], parse_dates=['date'], index_col='date')

# Print data
print(df)

# Visualize the time series
plt.plot(df.index, df['net sales'])
plt.xlabel('Year')
plt.ylabel('net sales')
plt.title('net sales Time Series')
plt.show()

try:
    plt.savefig('net_sales_plot.png')
except:
    print('Error saving plot')

plt.close('all')

# Decompose the time series into its trend, seasonality, and residual components
decomposition = seasonal_decompose(df['net sales'], model='multiplicative', period=12)

# Plot the decomposed components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(df['net sales'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

try:
    plt.savefig('decomposition_plot.png')
except:
    print('Error saving plot')

plt.close('all')

# Find the optimal ARIMA model parameters using the auto_arima function
print("Finding the optimal ARIMA model parameters...")
model = auto_arima(df['net sales'], seasonal=True, m=12, suppress_warnings=True)

# Print the summary of the ARIMA model parameters
print(model.summary())

# Split the data into training and testing sets
train_data = df.iloc[:-12]
test_data = df.iloc[-12:]

# Train the ARIMA model on the training data
print("Training the ARIMA model on the training data...")
model.fit(train_data['net sales'])

# Generate predictions for the testing period
print("Generating predictions for the testing period...")
predictions = model.predict(n_periods=12)

# Print the predictions
print('Predictions:', predictions)

# Evaluate the performance of the ARIMA model
print("Evaluating the performance of the ARIMA model...")
mae = mean_absolute_error(test_data['net sales'], predictions)
mse = mean_squared_error(test_data['net sales'], predictions)
rmse = np.sqrt(mse)
print('Mean Absolute Error:', mae)
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)

# Save the trained ARIMA model as a joblib file
joblib.dump(model, 'arima_model.joblib')

# Get the feature importance using SHAP
print("Getting the feature importance using SHAP...")
explainer = shap.Explainer(model)
shap_values = explainer.shap_values(test_data)

# Plot the summary plot
fig1 = shap.summary_plot(shap_values, test_data, plot_type="bar")
fig1.show()

# Plot the force plot for a single instance
instance = test_data.iloc[0]
shap_values_inst = explainer(instance)
fig2 = shap.force_plot(explainer.expected_value, shap_values_inst.values, instance)
fig2.show()
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go

def shap_dashboard(shap_values, data):
    # Create a scatter plot of the SHAP values
    shap_scatter = go.Scattergl(
        x = shap_values,
        y = data['net sales'],
        mode = 'markers',
        marker = dict(
            size = 5,
            opacity = 0.7,
            color = shap_values,
            colorscale = 'Viridis',
            showscale = True
        )
    )

    # Create a histogram of the SHAP values
    shap_hist = go.Histogram(
        x = shap_values,
        opacity = 0.7,
        marker=dict(color='purple')
    )

    # Create a layout for the dashboard
    layout = html.Div([
        html.H1('SHAP Dashboard'),
        dcc.Graph(id='shap-scatter', figure={'data': [shap_scatter], 'layout': {'title': 'SHAP Values Scatterplot'}}),
        dcc.Graph(id='shap-hist', figure={'data': [shap_hist], 'layout': {'title': 'SHAP Values Histogram'}})
    ])

    return layout

# Create a dashboard to visualize the SHAP values for all instances
app = JupyterDash(__name__)
app.layout = shap_dashboard(shap_values, test_data)
app.run_server(mode='inline', port=8050)

# Save the dashboard to a file
with open('shap_dashboard.html', 'w') as f:
    f.write(app._terminate_server_for_port("localhost", 8050))

# Define the parameter ranges to search over
p = range(0, 3)
d = range(0, 2)
q = range(0, 3)
P = range(0, 3)
D = range(0, 2)
Q = range(0, 3)

# Generate all possible combinations of parameters
param_combinations = list(itertools.product(p, d, q, P, D, Q))

# Define a function to train and evaluate a SARIMA model given the parameters
def evaluate_sarima(params, train_data, test_data):
    model = SARIMAX(train_data['net sales'], order=(params[0], params[1], params[2]), seasonal_order=(params[3], params[4], params[5], 12))
    model_fit = model.fit(disp=False)
    predictions = model_fit.forecast(steps=len(test_data))
    mae = mean_absolute_error(test_data['net sales'], predictions)
    return mae

# Split the data into training and testing sets
train_data = df.iloc[:-12]
test_data = df.iloc[-12:]

# Evaluate the performance of all parameter combinations using a grid search
best_params = None
best_mae = float('inf')
for params in param_combinations:
    try:
        mae = evaluate_sarima(params, train_data, test_data)
        if mae < best_mae:
            best_params = params
            best_mae = mae
    except:
        continue

# Train the best SARIMA model on the full dataset
model = SARIMAX(df['net sales'], order=(best_params[0], best_params[1], best_params[2]), seasonal_order=(best_params[3], best_params[4], best_params[5], 12))
model_fit = model.fit(disp=False)

# Generate predictions for the testing period
predictions = model_fit.forecast(steps=len(test_data))

# Evaluate the performance of the best SARIMA model
mae = mean_absolute_error(test_data['net sales'], predictions)
print(f"Best parameters: {best_params}")
print(f"MAE on test data: {mae}")
