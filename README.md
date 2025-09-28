
# GPU-Accelerated-Batch-Inference-Pipeline Project

For this project, I wanted to explore how we can improve the speed of working with big datasets using GPUs with tools like RAPIDS for preprocessing and Triton for model hosting and deployment.
The goal is to set a solid foundation for a fully optimized batch inference pipeline from data processing to model deployment.

---

## ✨ Features

* **Data preprocessing with RAPIDS cuDF & Dask-cuDF**
  GPU-accelerated data loading, feature engineering, and scaling.

* **Model training & export**
  Train an XGBoost model on GPU and export it in a Triton-compatible format.

* **Model serving with NVIDIA Triton**
  Deploy the trained model with Triton for scalable inference.

* **Batch inference pipeline**
  Process millions of rows in batches, send requests to Triton, and collect predictions.

* **MLOps-ready workflow**
  Containerized, reproducible, and scalable to multi-GPU / multi-node environments.
  (Future work: add model monitoring with Grafana and Prometheus).

---

## 🛠 Tech Stack

* [RAPIDS cuDF](https://rapids.ai/) – GPU-accelerated DataFrame library
* [Dask-cuDF](https://docs.rapids.ai/api/dask-cudf/stable/) – distributed multi-GPU data processing
* [XGBoost (GPU version)](https://xgboost.readthedocs.io/en/stable/gpu/index.html) – training the model
* [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) – scalable inference
* [Triton Python client](https://github.com/triton-inference-server/client) – send batch requests to Triton
* [Docker](https://www.docker.com/) – containerization

---

## 📂 Project Structure

```
├── datapreprocessing/
│   └── preprocess.py       # Preprocessing with RAPIDS
├── model_training/
│   ├── model.py            # Model training
│   └── inference.py        # Batch inference pipeline (RAPIDS + Triton client)
├── triton_model_repo/      # Model repository for Triton
├── docker-compose.yml      # Triton + pipeline setup
└── README.md               # Project documentation
```

---

## 📚 References

* [RAPIDS.ai](https://rapids.ai/)
* [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)
* [XGBoost GPU Support](https://xgboost.readthedocs.io/en/stable/gpu/index.html)


