# DRST: a Non-Intrusive Framework for Performance Analysis in Softwarized Networks

This repository supplements our [XX 2025 paper] titled  
**"DRST: a Non-Intrusive Framework for Performance Analysis in Softwarized Networks"**.  

It provides open-source infrastructure to study and reproduce the experiments presented in the paper, including data, scripts, and code for performance inference, forecasting, and drift-aware self-tuning in softwarized NFV environments.

---

## 📘 Overview
This repository is organized as follows:

- **drst_inference/**  
  Implementation of throughput and latency inference modules, including ML-based predictors and evaluation scripts.  

- **drst_forecasting/**  
  Time-series forecasting pipeline (MLP, LSTM, etc.) to predict performance evolution at short-term horizons.  

- **drst_drift/**  
  Drift detection and adaptive retraining logic, with SHAP-based feature sensitivity analysis.  

- **experiments/**  
  Scripts and datasets to reproduce the main results from our paper (inference accuracy, SoTA comparison, drift-adaptive retraining).  

- **docs/**  
  Documentation, figures, and experiment instructions.  

---

## 📊 Current limitations
- Current implementation is validated on **Kubernetes + Kubeflow** with **Intel Xeon (Cascade Lake)** servers.  
- Extensions to AMD platforms and other MLOps stacks are left as future work.  
- Lightweight online retraining is currently implemented for regression models; deep RL retraining logic will be released in future updates.  

---

## 📄 Technical Report
An extended version of our paper, with additional results and detailed methodology, is available here:  
👉 **[tech-report.pdf](./docs/tech-report.pdf)**  

---

## 🛠 Usage
The contents of this repository can be freely used for **research and education purposes**.  
If you use our tools or find our work helpful, please cite the following publication:

```bibtex
@article{liu2025drst,
  author    = {Qiong Liu and Jianke Lin and Leonardo Linguaglossa and Tianzhu Zhang.},
  title     = {DRST: a Non-Intrusive Framework for Performance Analysis in Softwarized Networks},
  journal   = {xx},
  year      = {2025}
}
