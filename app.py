import gradio as gr
import pickle
import pandas as pd
from model import features_diff, data
import logging

# Configure logging
logging.basicConfig(filename='var_model_debug.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

import gradio as gr
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(filename='var_model_debug.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

# Load the saved VAR model
with open('var_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Ensure you have the correct DataFrame used during model training
features = data[['Open', 'High', 'Low', 'Close']]
features_diff = features.diff().dropna()

# Function to make predictions using the loaded VAR model
def predict_var(steps):
    try:
        steps = int(steps)
        if steps <= 0:
            return "Please enter a positive number of steps."
        
        # Get the last lag order values from the model
        last_values = features_diff.values[-loaded_model.k_ar:]
        logging.debug(f"Last Values: {last_values}")

        # Make a forecast
        forecast = loaded_model.forecast(y=last_values, steps=steps)
        logging.debug(f"Forecast: {forecast}")

        # Convert the forecast to a DataFrame
        forecast_df = pd.DataFrame(forecast, columns=features.columns)
        logging.debug(f"Forecast DataFrame:\n{forecast_df}")

        # Inverse differencing to get the forecasted values
        forecast_df = forecast_df.cumsum() + features.iloc[-1]
        logging.debug(f"Inverse Differenced Forecast DataFrame:\n{forecast_df}")

        # Create plots for each feature
        plot_paths = []
        for column in forecast_df.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(forecast_df.index, forecast_df[column], label=f"Forecasted {column}")
            plt.title(f"Forecasted {column} for {steps} steps")
            plt.xlabel('Steps')
            plt.ylabel(column)
            plt.legend()
            plot_path = f"plot_{column}.png"
            plt.savefig(plot_path)
            plt.close()
            plot_paths.append(plot_path)

        return plot_paths

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return str(e)

# Create the Gradio interface
inputs = gr.inputs.Textbox(label='Number of Forecast Steps')
outputs = gr.outputs.Image(type="filepath", label='Forecasted Prices')

gr.Interface(fn=predict_var, inputs=inputs, outputs=outputs, title='Nifty 50 VAR Model Forecast').launch()