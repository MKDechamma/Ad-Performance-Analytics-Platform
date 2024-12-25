from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = 'Advertising.csv'
data = pd.read_csv(file_path)

# Select relevant columns
X = data[['Radio', 'TV', 'Newspaper']].values
y_sales_increase = data['%IncreseOnSales'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_sales_increase, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model_sales_increase=LinearRegression()
model_sales_increase.fit(X_train, y_train)

# Function to predict sales increase
def predict_sales_increase(Radio, TV, Newspaper):
    input_data = np.array([[Radio, TV, Newspaper]])
    predicted_increase = model_sales_increase.predict(input_data)
    return predicted_increase[0]

# Function to create scatter plot
def create_scatter_plot():
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data['TV'], y=data['%IncreseOnSales'], color='blue', label='TV')
    sns.scatterplot(x=data['Radio'], y=data['%IncreseOnSales'], color='red', label='Radio')
    sns.scatterplot(x=data['Newspaper'], y=data['%IncreseOnSales'], color='green', label='Newspaper')
    plt.xlabel('Advertising Expenditure')
    plt.ylabel('% Increase on Sales')
    plt.title('Scatter Plot of Advertising Expenditure vs Sales Increase')
    plt.legend()
    plt.grid(True)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return plot_data

# Function to create line plot
def create_line_plot():
    plt.figure(figsize=(10, 6))
    plt.plot(data['TV'], data['%IncreseOnSales'], marker='o', linestyle='-', color='blue', label='TV')
    plt.plot(data['Radio'], data['%IncreseOnSales'], marker='o', linestyle='-', color='red', label='Radio')
    plt.plot(data['Newspaper'], data['%IncreseOnSales'], marker='o', linestyle='-', color='green', label='Newspaper')
    plt.xlabel('Advertising Expenditure')
    plt.ylabel('% Increase on Sales')
    plt.title('Line Plot of Advertising Expenditure vs Sales Increase')
    plt.legend()
    plt.grid(True)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return plot_data

# Function to create histogram
def create_histogram(data, column_name):
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column_name], bins=20, kde=True)
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.title(f'Histogram of {column_name}')
    
    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return plot_data

# Flask app initialization
app = Flask(__name__)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/scatter_plot')
def scatter_plot():
    plot_data = create_scatter_plot()
    return render_template('scatter_plot.html', plot_data=plot_data)

@app.route('/line_plot')
def line_plot():
    plot_data = create_line_plot()
    return render_template('line_plot.html', plot_data=plot_data)

@app.route('/histogram')
def histogram():
    # Assuming df is your DataFrame with advertising spends and % increase on sales
    df = pd.DataFrame(X, columns=['Radio', 'TV', 'Newspaper'])
    df['%IncreaseOnSales'] = y_sales_increase

    radio_hist = create_histogram(df, 'Radio')
    tv_hist = create_histogram(df, 'TV')
    newspaper_hist = create_histogram(df, 'Newspaper')

    return render_template('histogram.html', radio_hist=radio_hist, tv_hist=tv_hist, newspaper_hist=newspaper_hist)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        radio = float(request.form['Radio'])
        tv = float(request.form['TV'])
        newspaper = float(request.form['Newspaper'])

        # Check for NaN values in inputs
        if np.isnan([radio, tv, newspaper]).any():
            raise ValueError("Input values must be numeric.")

        predicted_increase = predict_sales_increase(radio, tv, newspaper)

        if predicted_increase is None:
            raise ValueError("Failed to predict sales increase.")

        return render_template('index.html',
                               radio=radio,
                               tv=tv,
                               newspaper=newspaper,
                               predicted_increase=predicted_increase)
    except ValueError as ve:
        return render_template('index.html', error=str(ve))
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.static_folder = 'static'
    app.run(debug=True)
