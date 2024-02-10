# Startup Profit Prediction

This Python script analyzes startup data and predicts profit using a linear regression model. It includes data collection, preparation, visualization, model training, evaluation, and prediction.

## Description

The script performs the following steps:

- Collects startup data from a CSV file hosted on GitHub.
- Prepares the data by replacing categorical values with dummy variables.
- Visualizes the data distribution and correlation using seaborn.
- Splits the data into training and testing sets.
- Trains a linear regression model using the training data.
- Evaluates the model's performance using R^2 score and mean squared error.
- Predicts profit using the testing data and visualizes the results with a regression plot.

## Usage

1. Ensure you have Python installed on your system.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the script using `python startup_profit_prediction.py`.
4. View the generated visualizations and model evaluation metrics.
5. Analyze the prediction results plotted on the regression plot.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## License

This project is licensed under the MIT License.

## Author

Naresh R
