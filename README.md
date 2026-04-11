# Deployment-Oriented Operational Evaluation of Unsupervised IDS Models
Bachelor's thesis project evaluating unsupervised anomaly detection models for 
network intrusion detection in smart grid substations. Models are trained 
exclusively on normal IEC 61850 traffic and evaluated under explicit false 
positive rate (FPR) budgets to assess operational suitability for live deployment.

## Models
- Dense Autoencoder (Dense AE)
- LSTM Autoencoder (LSTM AE)
- Hybrid Autoencoder (Hybrid AE): ablation variant combining reconstruction 
  error with Mahalanobis distance in latent space
- Isolation Forest
- One-Class SVM

## Dataset
Network traffic was generated using the SGSim smart grid substation emulator, 
capturing IEC 61850 GOOSE and Sampled Values (SV) protocol traffic under normal 
operation and two attack families: Denial of Service (DoS) and False Data 
Injection (FDI). The preprocessed dataset is provided as 'all.parquet'.

## Repository Structure
```
Autoencoder_operational_IDS_eval.ipynb  # Main notebook
all.parquet                             # Preprocessed dataset
requirements.txt                        # Python dependencies
README.md
```

## Requirements
install with:
```bash
pip install -r requirements.txt
```

## How to run
Clone the repository and open 'Autoencoder_operational_IDS_eval.ipynb
Run all cells top to bottom. The dataset 'all.parquet' must be in the same directory as the notebook.

```bash
git clone https://github.com/axeman8/Deployment-Oriented-Operational-Evaluation-of-Unsupervised-IDS-Models.git
```
