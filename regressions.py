import pandas as pd

# Load the original Excel file
df = pd.read_excel("Worksheet 1.xlsx")

# --- MODEL 1: PREDICT ACCELERATION ---
def predict_acceleration(row):
    return (
        17.10
        + 0.491 * row["Cylinders"]
        - 1.538 * row["Valves/Cylinder"]
        - 0.001441 * row["Displacement [cc]"]
        - 1.235 * row["Turbo"]
        - 0.0260 * row["Power [hp]"]
        + 0.0118 * row["Max Torque [Nm]"]
        - 0.00018 * row["Length [mm]"]
        + 0.00140 * row["Weight [kg]"]
        + 0.00055 * row["Trunk Volume [l]"]
    )

# --- MODEL 2: PREDICT MAXIMUM SPEED ---
def predict_max_speed(row):
    return (
        107.3
        - 0.00226 * row["Displacement [cc]"]
        + 5.44 * row["Turbo"]
        + 0.3020 * row["Power [hp]"]
        + 0.023 * row["Max Torque [Nm]"]
        + 0.0179 * row["Length [mm]"]
        - 0.0193 * row["Weight [kg]"]
    )

# --- MODEL 3: PREDICT FUEL CONSUMPTION ---
def predict_fuel_consumption(row):
    return (
        0.95
        + 0.000975 * row["Displacement [cc]"]
        - 0.149 * row["Turbo"]
        + 0.01626 * row["Power [hp]"]
        - 0.0089 * row["Max Torque [Nm]"]
        + 0.00120 * row["Weight [kg]"]
        + 0.686 * row["Cylinders"]
    )

# --- MODEL 4: PREDICT PRICE ---
def predict_price(row):
    return (
        -19860
        - 8825 * row["Turbo"]
        + 357 * row["Power [hp]"]
        + 254 * row["Max Torque [Nm]"]
        - 17.2 * row["Weight [kg]"]
        - 9626 * row["Fuel Consumption [l/100 km]"]
        + 16.5 * row["Length [mm]"]
        - 70 * row["Trunk Volume [l]"]
    )

# Select rows with missing target values (the 5 test cars)
missing_df = df[df["Price [EUR]"].isnull()].copy()

# Step-by-step prediction for each model
missing_df["Acceleration [sec.]"] = missing_df.apply(predict_acceleration, axis=1)
missing_df["Maximum Speed [km/h]"] = missing_df.apply(predict_max_speed, axis=1)
missing_df["Fuel Consumption [l/100 km]"] = missing_df.apply(predict_fuel_consumption, axis=1)
missing_df["Price [EUR]"] = missing_df.apply(predict_price, axis=1)

# Update the original DataFrame
df.update(missing_df)

# Save the final predicted dataset
df.to_excel("Worksheet 1 - Final Predictions.xlsx", index=False)

print("🎉 All 4 models successfully applied. Output saved as 'Worksheet 1 - Final Predictions.xlsx'")
