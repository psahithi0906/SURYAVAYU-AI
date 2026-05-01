# SURYAVAYU-AI
SURYAVAYU AI is an AI-powered forecasting system for solar and wind energy generation across Karnataka's power grid, built for KREDL/KSPDCL. It uses a combination of LightGBM, LSTM, and Conformal Prediction models to deliver day-ahead and intra-day forecasts with probabilistic uncertainty bands (P10/P50/P90). The system is non-invasive, working as an overlay on existing SCADA/EMS infrastructure without any modifications. Every forecast includes SHAP-based explainability so operators understand the reasoning behind predictions. It targets an NRMSE below 10% and includes bonus features like auto-retraining, satellite nowcasting, and regulatory report generation


### PREREQUISITES :
Node==24
Python==3.8

### Commands to start:
```
docker-compose build 

docker-compose up -d 
```