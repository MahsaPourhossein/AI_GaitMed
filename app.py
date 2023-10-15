from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


app = Flask(__name__)

# Load the model using pickle when the app starts
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

scaler = joblib.load('scaler.pkl')

def get_mid_data(df):
    # Get the number of rows
    duration = len(df)

    # Get the middle point of duration
    mid = duration // 2
    start = mid - 750
    end = mid + 750

    # Get the data from the middle point
    df = df[start:end]
    return df

def convert_time(time):
    return float(time.split(':')[2])

def read_data(filename):
    # Read the csv file
    df = pd.read_csv(filename)
    
    # Get relevant data
    df = get_mid_data(df)
    
    # Convert time to float if Eamon not in filename
    df['time'] = df['time'].apply(convert_time)

    # Get the difference between each time
    df["diff"] = df["time"].diff()

    # First row has a NaN value for diff, so change it to 0.01
    df['diff'].iloc[0] = 0.01

    # Only take points that have a diff greater than 0
    df = df[df['diff'] > 0]

    # Create a new column called sample_times that start at 0.01 and increments by 0.01 for each row (0.01, 0.02, 0.03, etc.)
    df['sample_times'] = np.arange(0.01, len(df)*0.01+0.01, 0.01)

    # Convert this to an array of the structure [sample_time] = [ax, ay, az, wx, wy, wz]
    sample = df[['ax', 'ay', 'az', 'wx', 'wy', 'wz']].to_numpy()

    # Keep the first 1495 elements in sample
    sample = sample[:1495]

    # Return the numpy array
    return sample

def preprocess_data(file):
    sample = read_data(file)
    # Split sample into two samples
    sample = sample[750:]
    return sample

def normalise_data(data):
    # Reshape data to 2D array
    # Currently in shape (y, z), need to be (1, y*z)
    print(data.shape)
    X_reshaped = data.reshape(1, data.shape[0]*data.shape[1])
    print(X_reshaped.shape)
    x_standardised = scaler.transform(X_reshaped)
    return x_standardised

def create_plot(filename):
    df = pd.read_csv(filename)
    df.plot(figsize=(50,10))
    plt.savefig('static/plot.png')

def plot_all_axes_from_csv(filename):
    # Load the data from the CSV
    dataframe = pd.read_csv(filename)
    
    # Create a 3x2 grid for the 6 plots
    fig, ax = plt.subplots(3, 2, figsize=(15, 10))
    ax = ax.ravel()

    # Plot each axis
    axes = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']
    colours = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for index, axis in enumerate(axes):
        # If axis starts with a, set label to linear acceleration, else set it to angular velocity
        label = 'Angular Velocity'
        if "a" in axis:
            label = 'Linear Acceleration'
        ax[index].plot(dataframe[axis], color=f'tab:{colours[index]}')
        ax[index].set_title(f'{axis}')
        ax[index].set_xlabel('Time')
        ax[index].set_ylabel(f'{label}')
    
    # Save the plots as a single image
    plt.tight_layout()
    plt.savefig('static/combined_plot.png')





@app.route('/', methods=['GET', 'POST'])
def index():
    message = None
    prediction = None

    if request.method == 'POST':
        if 'file' not in request.files:
            message = 'No file part'
        file = request.files['file']
        if file.filename == '':
            message = 'No selected file'
        if file and not message:
            # Create a plot of the data
            create_plot(file)
            file.seek(0)

            # Plot all axes from the CSV
            plot_all_axes_from_csv(file)
            file.seek(0)
            
            data = preprocess_data(file)
            data = normalise_data(data)

            # Use the pre-loaded model
            pred = model.predict(data)[0]
            if pred == 0:
                prediction = 'Abnormal gait detected'
            if pred == 1:
                prediction = 'Normal gait detected'

    return render_template('index.html', prediction=prediction, message=message)

@app.route('/reset', methods=['GET', 'POST'])
def reset():
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True)
