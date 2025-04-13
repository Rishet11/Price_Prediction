import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive Agg

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from prophet import Prophet
import json
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

df = pd.read_csv('Future_price/Future_years.csv')

app = Flask(__name__)

# Set a seaborn style for better aesthetics
sns.set_theme(style="whitegrid")

# Update the figure size and DPI for better quality
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def create_plot():
    # Add padding around the plot
    plt.tight_layout(pad=3.0)
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def home():
    # Load regions from JSON file
    with open('Future_price/Regions.json', 'r') as f:
        regions = json.load(f)['Region_Names']
    return render_template('index.html', regions=regions)

@app.route('/predict', methods=['POST'])
def predict():
    region_name = request.form['region']
    row_idx = df[df['RegionName'] == region_name].index[0]
    
    # Extract data starting from the 6th column (index 5)
    data = pd.DataFrame({
        'Date': df.columns[5:],
        'Property_Value': df.iloc[row_idx, 5:].values
    })
    
    # Convert Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Sort by date
    data = data.sort_values('Date').reset_index(drop=True)
    
    m = Prophet()
    
    # Rename columns for Prophet
    prophet_data = data.rename(columns={
        'Date': 'ds',
        'Property_Value': 'y'
    })
    
    # Fit the model
    m.fit(prophet_data)
    
    # Create future dates dataframe
    future = m.make_future_dataframe(periods=3650)
    
    # Make predictions
    forecast = m.predict(future)
    
    # Create the main forecast plot with enhanced styling
    fig1 = m.plot(forecast)
    plt.title(f'Property Value Predictions - {region_name}', 
              fontsize=16, 
              fontweight='bold', 
              pad=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Property Value ($)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(['Historical Data', 'Forecast', 'Uncertainty Interval'], 
              loc='upper left', 
              fontsize=12,
              frameon=True,
              facecolor='white',
              edgecolor='none',
              shadow=True)
    plot_url1 = create_plot()

    # Create the components plot with enhanced styling
    fig2 = m.plot_components(forecast)
    for ax in fig2.get_axes():
        ax.set_title(ax.get_title(), fontsize=14, pad=10)
        ax.tick_params(labelsize=12)
    plt.tight_layout(pad=3.0)
    plot_url2 = create_plot()
    
    return render_template('plot.html', 
                         forecast_plot=plot_url1, 
                         components_plot=plot_url2,
                         region=region_name)

if __name__ == '__main__':
    app.run(debug=True)