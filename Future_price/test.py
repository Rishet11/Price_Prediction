import json
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

print(Data_Extract('Chicago, IL'))