
# GPU-Accelerated-Batch-Inference-Pipeline Project

For this project, I wanted to explore how we can improve the speed of working with big datasets using GPUs with tools like RAPIDS for preprocessing and Triton for model hosting and deployment.
The goal is to set a solid foundation for a fully optimized batch inference pipeline from data processing to model deployment.

---

## ✨ Features

* **Data preprocessing with RAPIDS cuDF & Dask-cuDF**
  GPU-accelerated data loading, feature engineering, and scaling.

* **Model training & export**
  Train an XGBoost model on GPU and export it in a Triton-compatible format.

* **Model serving with NVIDIA Triton** (it works then decide not to work)
  Deploy the trained model with Triton for scalable inference.

* **Batch inference pipeline**
  Process rows in batches, send requests to Triton, and collect predictions.

* **MLOps-ready workflow**  (Future work: add model monitoring with Grafana and Prometheus).

  Containerized, reproducible, and scalable to multi-GPU / multi-node environments.

---

## 🛠 Tech Stack

* [RAPIDS cuDF](https://rapids.ai/) – GPU-accelerated DataFrame library
* [Dask-cuDF](https://docs.rapids.ai/api/dask-cudf/stable/) – distributed multi-GPU data processing
* [XGBoost (GPU version)](https://xgboost.readthedocs.io/en/stable/gpu/index.html) – training the model
* [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) – scalable inference
* [Triton Python client](https://github.com/triton-inference-server/client) – send batch requests to Triton
* [Docker](https://www.docker.com/) – containerization
  
  PS : for the dataset I worked initially with
  https://www.kaggle.com/datasets/iamsouravbanerjee/airline-dataset
---

## 📂 Project Structure

```
├── rapidspreprocessing/
│   └── preprocess.py       # Preprocessing with RAPIDS
├── model_training/
│   ├── model.py            # Model training
│   └── inference.py       
├── triton_model_repo/      # (to add)  
└── README.md               
```

---

## 📚 References

* [RAPIDS.ai](https://rapids.ai/)
* [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)
* [XGBoost GPU Support](https://xgboost.readthedocs.io/en/stable/gpu/index.html)


