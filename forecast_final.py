import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clearml import Task
import pickle
from numerize import numerize


def get_dataframe(num_rows):

    task = Task.get_task('7b30f2f252e04d15be209f68d23f4d18')
    args = {'steps' : num_rows}
    model_path  = task.artifacts['sarimax_model'].get_local_copy()

    with open(model_path, 'rb') as handle:
        model = pickle.load(handle)
    res = model.forecast(steps=args['steps'])
    df = pd.DataFrame(res)
    df.reset_index(inplace=True)
    df.columns = ['timestamp','Sales' ]
    return df


st.set_page_config(layout="wide")
st.title('Retail Store Sales Forecast')

# Get user input
num_rows = st.number_input('Forecast for next:', min_value=1, max_value=1000, value=14)

# Generate the DataFrame
df = get_dataframe(num_rows)

dates = np.array(df["timestamp"].tolist())
sales = np.array(df["Sales"].tolist())

# Calculate the minimas and the maximas
minimas = (np.diff(np.sign(np.diff(sales))) > 0).nonzero()[0] + 1 
maximas = (np.diff(np.sign(np.diff(sales))) < 0).nonzero()[0] + 1

# Plot the entire data first
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(dates, sales)
# Then mark the maximas and the minimas
for minima in minimas:
    plt.plot(df.iloc[minima]["timestamp"], df.iloc[minima]["Sales"], marker="o")
    y_label = numerize.numerize(df.iloc[minima]['Sales'])
    ax.text(df.iloc[minima]["timestamp"], df.iloc[minima]["Sales"], f"{y_label}", size=12)
    
for maxima in maximas:
    plt.plot(df.iloc[maxima]["timestamp"], df.iloc[maxima]["Sales"], marker="o")
    y_label = numerize.numerize(df.iloc[maxima]['Sales'])
    ax.text(df.iloc[maxima]["timestamp"], df.iloc[maxima]["Sales"], f"{y_label}", size=12)

#plt.legend()
ax.set_title(f'Sales for next {num_rows} Days')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Sales')
plt.xticks(rotation=45)

# Display the plot
plot_column, table_column = st.columns([2, 1])
plot_column.pyplot(fig)

# Display the table with colored column headings
table_style = {'selector': 'th', 'props': [('background-color', '#eab676')]}
table_column.table(df.style.set_table_styles([table_style]))
