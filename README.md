# CM4371 – Sri Lankan Used Car Price Ensemble Learning Dashboard

An interactive Streamlit dashboard comparing **Bagging (Random Forest)**, **Boosting (Gradient Boosting & XGBoost)**, **Stacking (RF + XGB + GBR → Ridge)**, and a **Hybrid (RF + ANN Residual Learning)** model on 9,788 real Sri Lankan used car listings.

### Key Results (Test Set)
| Model | RMSE (LKR M) | MAE (LKR M) | R² Score | Train Time |
| :--- | :---: | :---: | :---: | :---: |
| **Hybrid (RF + ANN)** | **1.219** | 0.594 | **0.9143** | 16.62s |
| **Bagging (Random Forest)** | 1.241 | **0.576** | 0.9112 | 22.58s |
| **Stacking (RF + XGB + GBR)** | 1.262 | 0.603 | 0.9081 | 145.64s |
| **Boosting (XGBoost)** | 1.285 | 0.611 | 0.9048 | **1.61s** |
| **Boosting (Gradient Boosting)** | 1.307 | 0.664 | 0.9015 | 23.47s |

### Generative AI Disclosure
Generative AI (Gemini) was used to assist with Python boilerplate code, Streamlit UI layout, and structuring the comparison pipeline.