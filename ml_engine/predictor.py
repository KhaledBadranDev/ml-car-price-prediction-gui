# ml_engine/predictor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class CarPricePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.training_columns = None
        self.unique_values = {} # To store dropdown options for the GUI
        
    def load_and_train(self):
        # Load Data
        data_frame = pd.read_csv(self.data_path)
        
        # --- Pre-processing for GUI Dropdowns ---
        # We save unique values before encoding to populate the GUI lists later
        cat_cols = data_frame.select_dtypes(include=['object']).columns
        for col in cat_cols:
            self.unique_values[col] = sorted(data_frame[col].unique().tolist())
            
        # Also grab min/max for numerical fields if needed (optional)
        self.unique_values['year'] = sorted(data_frame['year'].unique().tolist(), reverse=True)

        # --- Standard ML Pipeline from your main.py ---
        y = data_frame["selling_price"].apply(lambda x: np.log1p(x))
        X = data_frame.drop(columns=["selling_price"])
        
        # One-Hot Encoding
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
        # Save the column names! 
        # Crucial: User input must be re-indexed to match these exact columns.
        self.training_columns = X.columns
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Evaluate (Optional log to console)
        self._evaluate(X_test, y_test)
        print("Model Trained Successfully.")

    def _evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"Model Evaluation -> MSE: {mse:.4f}, R2: {r2:.4f}")

    def predict_price(self, input_data: dict):
        """
        input_data: dict containing 'name', 'year', 'km_driven', 'fuel', etc.
        """
        # 1. Convert input dict to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # 2. One-Hot Encode the input using the same method
        # Note: We must know which columns were categorical
        cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
        input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
        
        # 3. Align Columns
        # The input_df will likely have fewer columns than the trained model 
        # (because the user only selected 1 car name, not all 1000).
        # We reindex to add missing columns with 0.
        input_df = input_df.reindex(columns=self.training_columns, fill_value=0)
        
        # 4. Predict
        log_price = self.model.predict(input_df)[0]
        
        # 5. Inverse Log Transformation (np.expm1) to get actual currency
        actual_price = np.expm1(log_price)
        
        return round(actual_price, 2)