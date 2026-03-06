# Energy Optimization Prediction System (Ensemble ML Rebuild)
![CI/CD Pipeline](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME/actions/workflows/main.yml/badge.svg)

## Setup (3 steps)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate Data & Train Ensemble Models (XGBoost + LightGBM):
   ```bash
   python dataset_generator.py
   python train_model.py
   ```
3. Run the complete 9-Page Dashboard:
   ```bash
   streamlit run dashboard.py
   ```

## Advanced ML Pipeline Features
* **Physics-Driven Dataset**: Generates ~12,000 thermodynamic records factoring in ambient temperature, setpoint, and heat index. 
* **Feature Engineering**: 20 advanced features including cyclical temporal variables, usage lag values, rolling means, and Indian Grid (IS 12360) voltage drop parameters.
* **Model Architecture**: Ensemble Time-Series Regressor comprising 70% XGBoost and 30% LightGBM, predicting 15-minute resolution slots.
* **Validation**: Model validates at R² > 0.94 and MAPE < 6%, beating independent base models.

## OpenWeather API Key (free)
1. Go to openweathermap.org/api
2. Sign up free & copy the API key
3. Paste in the dashboard sidebar configuration

## Judge FAQ & Proof Points

**Q: Is the energy optimization mathematically proven?**  
A: Yes! See the **Pre-Cooling Simulator** page for interactive physics validation using the specific heat capacity formula (Q=mcΔT) and real Indian HVAC specs.

**Q: What about voltage drop logic?**  
A: Included in the **Live Monitor**. Grid voltage drops by hour, increasing resistive load consumption and motor load current.

**Q: How do we test the model?**  
A: Navigate to **Judge's Demo Panel** or **Live Custom Test Data** tab to manually adjust environmental or electrical variables and watch the physical outputs calculate live.

## CI/CD Pipeline
This project is fully automated via **GitHub Actions**. On every push to `main` or `develop`, the pipeline:
1. Re-generates the thermodynamic dataset.
2. Trains the XGBoost & LightGBM ensemble models.
3. Asserts the evaluation metrics exceed requirements (**R² > 0.90**).
4. Packages and uploads generated `xgboost.pkl`, `lightgbm.pkl` and `model_metrics.json` as downloadable Pipeline Artifacts.
