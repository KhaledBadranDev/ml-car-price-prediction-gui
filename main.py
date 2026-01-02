# main.py
import sys
import os

# Ensure Python can find our custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from ml_engine.predictor import CarPricePredictor
from gui.app_window import CarPriceApp

def main():
    print("Initializing Machine Learning Engine...")
    
    # 1. Initialize Predictor
    # Assuming cars-data.csv is in a folder named 'data'
    data_file = './data/cars-data.csv' 
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    predictor = CarPricePredictor(data_file)
    
    # 2. Train Model
    print("Training model... Please wait.")
    predictor.load_and_train()
    
    # 3. Launch GUI
    print("Launching GUI...")
    app = CarPriceApp(predictor)
    app.mainloop()

if __name__ == "__main__":
    main()