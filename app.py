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
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load models and data
df_future = pd.read_csv('Future_price/Future_years.csv')

try:
    with open('price_pred/house_price_model.pkl', 'rb') as model_file:
        house_price_model = pickle.load(model_file)
    with open('price_pred/model_features.pkl', 'rb') as features_file:
        feature_names = pickle.load(features_file)
except Exception as e:
    print(f"Error loading model or features: {str(e)}")
    raise

# Set plotting configurations
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 16]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def create_plot():
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9)
    plt.tight_layout(pad=3.0)
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def predict_property_price(model, rental_yield, appreciation_rate, crime_rate, aqi, 
                         transport_score, school_rating, walkability, city):
    cities = {
        'Boston': 0, 'Chicago': 0, 'Dallas': 0, 'Denver': 0,
        'Houston': 0, 'Los Angeles': 0, 'Miami': 0, 'New York City': 0,
        'San Francisco': 0, 'Seattle': 0
    }
    
    if city in cities:
        cities[city] = 1
    
    input_data = np.array([[
        rental_yield,
        appreciation_rate,
        crime_rate,
        aqi,
        transport_score,
        school_rating,
        walkability,
        *cities.values()
    ]])
    
    prediction = model.predict(input_data)
    return prediction[0]

# Routes for future price prediction
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/future')
def future_home():
    with open('Future_price/Regions.json', 'r') as f:
        regions = json.load(f)['Region_Names']
    return render_template('future/index.html', regions=regions)

@app.route('/future/predict', methods=['POST'])
def predict_future():
    region_name = request.form['region']
    row_idx = df_future[df_future['RegionName'] == region_name].index[0]
    
    data = pd.DataFrame({
        'Date': df_future.columns[5:],
        'Property_Value': df_future.iloc[row_idx, 5:].values
    })
    
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    
    m = Prophet()
    prophet_data = data.rename(columns={'Date': 'ds', 'Property_Value': 'y'})
    m.fit(prophet_data)
    future = m.make_future_dataframe(periods=3650)
    forecast = m.predict(future)
    
    fig1 = m.plot(forecast)
    plt.title(f'Property Value Predictions - {region_name}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Years', fontsize=14, labelpad=10)
    plt.ylabel('Property Value ($)', fontsize=14, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.legend(['Historical Data', 'Forecast', '95% Confidence Interval'], 
              loc='upper left', fontsize=12, frameon=True,
              facecolor='white', edgecolor='none', shadow=True)
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9)
    plot_url1 = create_plot()

    fig2 = m.plot_components(forecast)
    for ax in fig2.get_axes():
        ax.set_title(ax.get_title(), fontsize=14, pad=10)
        ax.tick_params(labelsize=12)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=7))
        ax.set_position([0.1, ax.get_position().y0, 0.8, ax.get_position().height])
    plt.tight_layout(pad=3.0)
    plot_url2 = create_plot()
    
    return render_template('future/plot.html', 
                         forecast_plot=plot_url1, 
                         components_plot=plot_url2,
                         region=region_name)

# Routes for house price prediction
@app.route('/price')
def price_home():
    return render_template('price/index.html', feature_names=feature_names)

@app.route('/price/predict', methods=['POST'])
def predict_price():
    try:
        input_data = []
        for feature in feature_names[:7]:
            value = float(request.form[feature])
            input_data.append(value)

        city = request.form['city']
        prediction = predict_property_price(
            model=house_price_model,
            rental_yield=input_data[0],
            appreciation_rate=input_data[1],
            crime_rate=input_data[2],
            aqi=input_data[3],
            transport_score=input_data[4],
            school_rating=input_data[5],
            walkability=input_data[6],
            city=city
        )
        
        formatted_prediction = f"${prediction:,.2f}"
        return render_template('price/index.html', 
                             feature_names=feature_names,
                             prediction=formatted_prediction)

    except Exception as e:
        return render_template('price/index.html', 
                             feature_names=feature_names,
                             prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)