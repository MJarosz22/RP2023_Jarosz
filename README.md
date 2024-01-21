# Pathway Optimization with GAN and PPCA

A project for pathway optimization using Generative Adversarial Networks (GAN) and Probabilistic Principal Component Analysis (PPCA). This project aims to generate synthetic data for pathway optimization using both GAN and PPCA generative models.



## Introduction

This project utilizes Generative Adversarial Networks (GAN) and Probabilistic Principal Component Analysis (PPCA) to generate synthetic data for pathway optimization. It includes implementations of GAN and PPCA models, along with a training loop and synthetic data generation.

## Features

- GAN model for synthetic data generation
- PPCA model for synthetic data generation
- Visualization of data in the original dimensions
- Visualization of data in PCA reduced dimensions
- Visualization of how generated data changes (and gets more realistic) throughout the training (for GAN only)
- Calculation of KL divergence between real and synthetic data

## Data
The data used for training can be accessed under the data directory. It consists of 5000 kinetic models, each representing a hypothetical pathway of E. Coli strain (generated and provided by supervisor of the project, MSc. Paul van Lent).

## Requirements

- Python 3.11
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Usage
An example usage - how to train models, generate data with them, visualize it - can be found in the Pathway_Optimization.ipynb notebook

## Acknowledgements

- Thanks to Oliver K. Ernst, PhD for the insightful [Medium article](https://medium.com/practical-coding/the-simplest-generative-model-you-probably-missed-c840d68b704) on PPCA, which was a basis for PPCA implementation in this project.

### Thesis Acknowledgement

This project was developed as part of the Bachelor's thesis at Delft University of Technology. The author would like to acknowledge the support and guidance received during this academic endeavor.
The thesis and other academic materials can be found in academic_work directory.

