# DriftNDiffuse: Latent Stochastic Differential Equations for Chaotic Systems

[![arXiv](https://img.shields.io/badge/arXiv-2606.xxxxx-B31B1B.svg)](https://arxiv.org/)
[![Zenodo](https://img.shields.io/badge/Zenodo-Data%20%26%20Weights-blue.svg)](https://zenodo.org/records/21037510?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI5OTU4MDRkLTM0MWYtNGU4OC05MTAxLTc5NzQ4OGY0NzZiYyIsImRhdGEiOnt9LCJyYW5kb20iOiI5NjU3NjQ0YzM3ZmE3OGI5NjJjNzQwOWFlZjFkNDIzYSJ9.E0uy6e9_llzMd3hODxP4NOme_q6aQiEfL8cpCCq0gStBZJrtWAlJqHjfBHYJG7h4UM9tL20SpJQOSxpvbPO2qQ)

Official PyTorch implementation for the paper: **"Modelisation of chaotic systems with a latent Stochastic Differential Equation"**.

This repository provides a non-intrusive, probabilistic Reduced-Order Model (ROM) framework tailored for chaotic dynamical systems. By projecting high-dimensional nonlinear dynamics (such as turbulent flows) onto a low-dimensional manifold via an autoencoder, the temporal evolution is explicitly tracked inside a latent space governed by a Stochastic Differential Equation (SDE). 

---

## 🚀 Key Features

* **Latent SDE Core:** Learns a deterministic *drift* term for predictable dynamics alongside a state-dependent *diffusion* term to capture underlying sub-grid uncertainties.
* **Manifold Preservation:** Generates unique, multi-admissible future chaotic trajectories that rigorously match the true transition kernels from Direct Numerical Simulation (DNS) data.
* **End-to-End Pipeline:** Complete training workflows, data preprocessing, and evaluation metrics (including Wasserstein distance tracking).

---

## 💾 Dataset & Pre-trained Weights

The baseline **Kuramoto-Sivashinsky** benchmark dataset along with our fully pre-trained model checkpoints are openly hosted on Zenodo.

📥 **[Download Data & Weights on Zenodo](https://zenodo.org/records/21037510?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI5OTU4MDRkLTM0MWYtNGU4OC05MTAxLTc5NzQ4OGY0NzZiYyIsImRhdGEiOnt9LCJyYW5kb20iOiI5NjU3NjQ0YzM3ZmE3OGI5NjJjNzQwOWFlZjFkNDIzYSJ9.E0uy6e9_llzMd3hODxP4NOme_q6aQiEfL8cpCCq0gStBZJrtWAlJqHjfBHYJG7h4UM9tL20SpJQOSxpvbPO2qQ)**

> 📌 *Note: Place downloaded `.npy` into your configured local data folder before executing training routines.*

---

## 🛠️ Installation

Ensure you have an active Python environment (Python >= 3.10 recommended). Clone the repository and install the optimized minimal dependencies layout:

```bash
git clone [https://github.com/IsmaelZig-SU/DriftNDiffuse.git](https://github.com/IsmaelZig-SU/DriftNDiffuse.git)
cd DriftNDiffuse
pip install -r requirements.txt```
