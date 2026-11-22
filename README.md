# Delhi 3-day Tmax and Heatwave Forecast (XGBoost)

Folder name must stay `heatwave-delhi` (in Google Drive: My Drive/heatwave-delhi).

## Run order (Google Colab)
1. Upload this whole folder to My Drive.
2. Open `notebooks/01_data_prep.ipynb` in Colab -> Run all (Drive mounts automatically). Raw data already exists, so no re-download.
3. Open `notebooks/02_model_training.ipynb` -> Run all (creates train/validation/test CSVs, configs, src/ modules).
4. Open `notebooks/03_xgboost_training.ipynb` -> Run all (trains XGBoost, creates models/ and reports/model/).
5. Download the folder from Drive to the laptop.

## Run the dashboard (laptop, VS Code terminal inside heatwave-delhi)
    pip install -r requirements.txt
    streamlit run app.py

`archive_old_hourly_classifier/` = the earlier hourly risk-class project (kept for reference, not used).
