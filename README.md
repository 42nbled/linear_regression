# Linear Regression

This project implements a simple linear regression model to predict car prices based on mileage. It provides hands-on experience with machine learning fundamentals, including data processing, optimization, and result interpretation.

---

## Features

- **Training:**  
  - Uses gradient descent to optimize parameters (`theta0` and `theta1`) for the linear regression model.
  - Reads training data from CSV files.
  - Saves the learned parameters to `parameters.txt`.

- **Prediction:**  
  - Loads the trained parameters and predicts car prices for given mileage values.
  - Can be used to make predictions on new data.

- **Data Visualization:**  
  - Plots the dataset and the regression line for better understanding and evaluation of the model.

---

## Project Structure

```
linear_regression/
├── data.csv
├── parameters.txt
├── predict.py
├── README.md
├── training.py
└── data/
    ├── data.csv
    ├── data2.csv
    └── data3.csv
```

- `training.py`: Script to train the linear regression model using gradient descent.
- `predict.py`: Script to predict car prices using the trained model.
- `parameters.txt`: Stores the learned parameters (`theta0` and `theta1`).
- `data.csv`, `data/`: Training and test datasets.
- `README.md`: Project documentation.

---

## Usage

### 1. Training the Model

Train the model using your dataset:

```sh
python3 training.py
```

- This will read data from `data.csv` (or another specified file), train the model, and save the parameters to `parameters.txt`.

### 2. Making Predictions

Predict car prices using the trained model:

```sh
python3 predict.py
```

- This will prompt for mileage input and output the predicted price using the parameters from `parameters.txt`.

---

## Requirements

- Python 3.x
- numpy
- matplotlib (for visualization)

Install dependencies with:

```sh
pip install numpy matplotlib
```

---

## Data

- The main dataset is `data.csv`. Additional datasets are available in the `data/` directory.
- Each CSV should contain two columns: mileage and price.

---

## Notes

- The project demonstrates the basics of linear regression and gradient descent.
- Visualization helps to interpret the fit and quality of the model.
- You can experiment with different datasets by modifying or adding files in the `data/` directory.
