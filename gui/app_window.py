import customtkinter as ctk
import tkinter.messagebox as msg_box  # Standard library for popups
from ml_engine.predictor import CarPricePredictor

# Set theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class CarPriceApp(ctk.CTk):
    def __init__(self, predictor: CarPricePredictor):
        super().__init__()

        self.predictor = predictor
        
        # Window Setup
        self.title("Car Price Predictor AI")
        self.geometry("900x700")
        
        # Title
        self.label_title = ctk.CTkLabel(self, text="Used Car Price Estimator", font=("Roboto Medium", 24))
        self.label_title.pack(pady=20)

        # Scrollable Frame
        self.scroll_frame = ctk.CTkScrollableFrame(self, width=800, height=400)
        self.scroll_frame.pack(pady=10, padx=20, fill="x", expand=False)

        # Dictionary to hold our input widgets
        self.inputs = {}
        
        # --- Create Inputs ---
        # 1. Car Name
        self.create_dropdown("Car Model", "name", self.predictor.unique_values.get('name', []))
        
        # 2. Year (Convert numbers to strings for the dropdown)
        years = [str(y) for y in self.predictor.unique_values.get('year', [])]
        self.create_dropdown("Year of Purchase", "year", years)
        
        # 3. Km Driven (Manual Entry)
        self.create_entry("Kilometers Driven", "km_driven")
        
        # 4. Fuel
        self.create_dropdown("Fuel Type", "fuel", self.predictor.unique_values.get('fuel', []))
        
        # 5. Seller Type
        self.create_dropdown("Seller Type", "seller_type", self.predictor.unique_values.get('seller_type', []))
        
        # 6. Transmission
        self.create_dropdown("Transmission", "transmission", self.predictor.unique_values.get('transmission', []))
        
        # 7. Owner
        self.create_dropdown("Previous Owners", "owner", self.predictor.unique_values.get('owner', []))

        # --- Predict Button ---
        self.btn_predict = ctk.CTkButton(
            self, 
            text="Predict Price", 
            command=self.on_predict, 
            height=50, 
            font=("Roboto Medium", 16)
        )
        self.btn_predict.pack(pady=20)

        # --- Result Label ---
        self.label_result = ctk.CTkLabel(
            self, 
            text="Estimated Price: ---", 
            font=("Roboto Medium", 20), 
            text_color="#00bba7"
        )
        self.label_result.pack(pady=10)

    def create_dropdown(self, label_text, key, values):
        container = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        container.pack(fill="x", pady=10)
        
        lbl = ctk.CTkLabel(container, text=label_text, anchor="w", width=200)
        lbl.pack(side="left", padx=10)
        
        # Safe default: if list is empty, use a placeholder
        default_val = str(values[0]) if values else "N/A"
        
        dropdown = ctk.CTkOptionMenu(container, values=[str(v) for v in values])
        dropdown.set(default_val)
        dropdown.pack(side="left", fill="x", expand=True, padx=10)
        
        self.inputs[key] = dropdown

    def create_entry(self, label_text, key):
        container = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        container.pack(fill="x", pady=10)
        
        lbl = ctk.CTkLabel(container, text=label_text, anchor="w", width=200)
        lbl.pack(side="left", padx=10)
        
        entry = ctk.CTkEntry(container, placeholder_text="e.g. 50000")
        entry.pack(side="left", fill="x", expand=True, padx=10)
        
        self.inputs[key] = entry

    def on_predict(self):
        # 1. Update text to show processing
        self.btn_predict.configure(text="Calculating...", state="disabled")
        self.update() # Force UI update immediately

        data = {}
        try:
            # print("Collecting input data...")  # Debug statement
            for key, widget in self.inputs.items():
                if isinstance(widget, ctk.CTkOptionMenu):
                    val = widget.get()
                    # Convert Year back to integer because the model expects a number
                    if key == "year":
                        data[key] = int(val)
                    else:
                        data[key] = val
                    # print(f"Collected dropdown value for {key}: {val}")  # Debug statement
                
                elif isinstance(widget, ctk.CTkEntry):
                    val = widget.get()
                    # Remove commas (e.g. "50,000" -> "50000") and spaces
                    clean_val = val.replace(",", "").replace(" ", "")
                    
                    if not clean_val.isdigit():
                        raise ValueError(f"The value for '{key}' must be a number.")
                    
                    data[key] = int(clean_val)
                    # print(f"Collected entry value for {key}: {clean_val}")  # Debug statement
            
            # 2. Call the Predictor
            # print("Calling predictor with data:", data)  # Debug statement
            price = self.predictor.predict_price(data)
            
            # 3. Success Update
            # print(f"Prediction successful. Estimated price: {price:,.2f}")  # Debug statement
            self.label_result.configure(text=f"Estimated Price: {price:,.2f}", text_color="#00bba7")
            self.update_idletasks()  # Force GUI update
            self.update()  # Force GUI update
            
        except Exception as e:
            # 4. Error Handling
            # Print to console for detailed debug
            # print("ERROR DETAILS:", e)
            import traceback
            traceback.print_exc()
            
            # Show Popup to user
            msg_box.showerror("Prediction Error", f"An error occurred:\n{str(e)}")
            self.label_result.configure(text="Error occurred", text_color="red")
            
        finally:
            # Reset button state
            # print("Resetting button state.")  # Debug statement
            self.btn_predict.configure(text="Predict Price", state="normal")