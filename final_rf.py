import pandas as pd
import numpy as np
import joblib
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

from google.oauth2 import service_account
from googleapiclient.discovery import build

class RainPredictionSystem:
    def __init__(self, credentials_path, spreadsheet_id, sheet_name):
        self.credentials_path = r'C:\Users\91982\Desktop\final year project\credentials.json'
        self.spreadsheet_id = 'https://docs.google.com/spreadsheets/d/1JXxsLUMpwkubsGcZ1iGEQuKhQxvrLB92JehDwILTB_4'
        self.sheet_name = sheet_name
        self.data = None
        self.rf_model = None
        self.scaler = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.future_df = None
        self.future_predictions = None

    def load_and_prepare_data(self):
        creds = service_account.Credentials.from_service_account_file(
    "credentials.json", scopes=["https://www.googleapis.com/auth/spreadsheets"]
                )

        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()

        result = sheet.values().get(spreadsheetId=self.spreadsheet_id,
                                    range=self.sheet_name).execute()
        values = result.get('values', [])

        if not values:
            raise Exception("❌ No data found in the Google Sheet.")

        headers = values[0]
        data = values[1:]

        self.data = pd.DataFrame(data, columns=headers)
        self.data[["temperature", "humidity", "pressure", "dew_point"]] = self.data[["temperature", "humidity", "pressure", "dew_point"]].astype(float)
        self.data["rain"] = self.data["rain"].astype(int)
        self.data["time"] = pd.to_datetime(self.data["time"])

        # Resample to balance dataset
        df_majority = self.data[self.data["rain"] == 0]
        df_minority = self.data[self.data["rain"] == 1]
        df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
        df_downsampled = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)

        base_X = df_downsampled.drop(columns=["rain", "time"], errors='ignore')
        trend_X = np.array([self.get_trend_features(index=i) for i in range(len(base_X))])
        full_X = np.hstack((base_X.values, trend_X))

        self.Y = df_downsampled["rain"]
        self.scaler = MinMaxScaler()
        full_X_scaled = self.scaler.fit_transform(full_X)

        self.X = full_X
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(full_X_scaled, self.Y, test_size=0.2, random_state=42)

    def train_model(self):
        self.rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, class_weight="balanced")
        self.rf_model.fit(self.X_train, self.Y_train)
        print("✅ Model and scaler trained successfully!")

    def get_trend_features(self, steps=5, index=None):
        if index is None or index < steps:
            index = len(self.data) - 1

        trend_features = []
        for col in ["temperature", "humidity", "pressure", "dew_point"]:
            values = self.data[col].values[max(0, index - steps + 1):index + 1]
            if len(values) < steps:
                values = np.pad(values, (steps - len(values), 0), mode='edge')
            trend = np.polyfit(range(steps), values, 1)[0]  # slope
            trend_features.append(trend)
        return np.array(trend_features)

    def predict_manual_input(self, temperature, humidity, pressure, dew_point):
        last = self.data.iloc[-1]
        trends = [temperature - last["temperature"], humidity - last["humidity"],
                  pressure - last["pressure"], dew_point - last["dew_point"]]
        inputs = np.hstack(([temperature, humidity, pressure, dew_point], trends))
        scaled = self.scaler.transform([inputs])
        proba = self.rf_model.predict_proba(scaled)[0]
        return np.argmax(proba), f"{proba[np.argmax(proba)] * 100:.1f}% chances"

    def predict_future(self):
        future_data = []
        for column in ["temperature", "humidity", "pressure", "dew_point"]:
            y = self.data[column].values[-48:]
            X_time = np.arange(len(y)).reshape(-1, 1)

            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(X_time)

            model = LinearRegression()
            model.fit(X_poly, y)

            future_X_time = np.arange(len(y), len(y) + 5).reshape(-1, 1)
            future_X_poly = poly.transform(future_X_time)

            predicted_values = model.predict(future_X_poly)

            if "humidity" in column:
                predicted_values = np.clip(predicted_values, 20, 100)
            if "pressure" in column:
                predicted_values = np.clip(predicted_values, 900, 1100)

            future_data.append(predicted_values)

        future_data = np.array(future_data).T
        self.future_df = pd.DataFrame(future_data, columns=["temperature", "humidity", "pressure", "dew_point"])

        future_trends = self.get_trend_features().reshape(1, -1)
        future_enhanced = np.hstack((self.future_df.values, np.repeat(future_trends, len(self.future_df), axis=0)))

        future_df_scaled = self.scaler.transform(future_enhanced)
        self.future_predictions = self.rf_model.predict(future_df_scaled)

        for i, (pred, row) in enumerate(zip(self.future_predictions, self.future_df.values)):
            next_time = pd.to_datetime(self.data["time"].iloc[-1]) + pd.Timedelta(days=i + 1)
            print(f"📅 Prediction for {next_time.date()}:")
            for col_name, value in zip(self.future_df.columns, row):
                print(f"  {col_name}: {value:.2f}")
            print(f"  🌧️ Rainfall Prediction: {'Yes' if pred == 1 else 'No'}\n")

    def evaluate_model(self):
        Y_pred = self.rf_model.predict(self.X_test)
        accuracy = accuracy_score(self.Y_test, Y_pred)
        print(f"✅ Model Accuracy: {accuracy * 100:.2f}%\n")
        print("🔹 Classification Report:\n", classification_report(self.Y_test, Y_pred))

    def get_recent_data_for_plot(self, count=100):
        # Safely get the last 'count' rows
        plot_data = self.data[["time", "temperature", "humidity", "pressure"]].copy()
        plot_data["time"] = pd.to_datetime(plot_data["time"])
        return plot_data.tail(count)



# ==== Run Everything ====

CREDENTIALS_PATH = r'C:\Users\91982\Desktop\final year project\credentials.json'  # 🔁 Update path
SPREADSHEET_ID = "https://docs.google.com/spreadsheets/d/1JXxsLUMpwkubsGcZ1iGEQuKhQxvrLB92JehDwILTB_4"                       # 🔁 Replace with your actual Google Sheet ID
SHEET_NAME = 'Sheet1'                                              # 🔁 Replace if your sheet name differs

system = RainPredictionSystem(CREDENTIALS_PATH, SPREADSHEET_ID, SHEET_NAME)
system.load_and_prepare_data()
system.train_model()
#system.predict_future()
system.evaluate_model()

# Save the full object
joblib.dump(system, "full_rain_prediction_system.pkl")
print("✅ Full system saved with all functionality!")
