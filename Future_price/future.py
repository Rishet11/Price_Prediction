import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Future_price/Future_years.csv')


Years= df.drop(['RegionID','SizeRank','RegionName','StateName'], axis=1)


Entries= Years.drop('RegionType', axis=1)



#Filter specific region
def find_region_rows(region_name):
    region_data = df[df['RegionName'] == region_name]
    return region_data.index.tolist()

"""## Graph (2000-2024)"""

def Graph(df, region_name):
    row= find_region_rows(region_name)
    X=df.iloc[row, :]


    # Example: Suppose df has 1 row and multiple columns, each column is a date
    # and the single row holds property prices.

    # Step 1: Transpose the DataFrame so columns become rows
    df_t = X.T  # Now each row is a (date, price) pair

    # Step 2: Rename the columns for clarity
    df_t = df_t.reset_index()
    df_t.columns = ["date", "price"]  # date is the old column header, price is the old row value

    # Step 3: Convert the date column to datetime format
    df_t["date"] = pd.to_datetime(df_t["date"], format="%Y-%m-%d")

    # Step 4: Plot using Pandas
    df_t.plot(x="date", y="price", figsize=(10, 6), marker="o", title="Property Price Over Time")

    # Step 5: Show the plot
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

Graph(Entries, 'United States')

"""## **Model Creation**

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

def Data_Extract(region_name):
    # Find row index for the region
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

    return data

# Test the function
Prices = Data_Extract('United States')

# Optional: Add basic validation
# print(f"\nShape of extracted data: {Prices.shape}")
# print(f"Date range: {Prices['Date'].min()} to {Prices['Date'].max()}")

def pred(df, region_name):
    data = Data_Extract(region_name)

    # Create Prophet model
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
    m.plot_components(forecast)

    # Visualize results
    fig = m.plot(forecast)
    plt.title(f'Property Value Predictions - {region_name}')  # Move title inside function
    plt.xlabel('Date')
    plt.ylabel('Property Value ($)')
    plt.tight_layout()
    plt.show()

    return m, forecast

# Test the function
# model, predictions = pred(df, 'United States')
