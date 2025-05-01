
# Fully Homomorphic Encrypted MRI Image Classification Comparison

This project explores the application of Fully Homomorphic Encryption (FHE) in privacy-preserving machine learning for medical image data, specifically focusing on MRI images. It compares the performance and accuracy of machine learning models operating on encrypted data versus plaintext data.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Processing Encrypted MRI Images](#processing-encrypted-mri-images)
  - [Processing Plain MRI Images](#processing-plain-mri-images)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

The repository contains Python implementations that demonstrate the feasibility of performing machine learning tasks on encrypted MRI images using FHE. It includes scripts for both encrypted and plaintext processing, allowing for a direct comparison in terms of performance and accuracy.

## Features

- Processing of MRI images using Fully Homomorphic Encryption.
- Comparison between encrypted and plaintext data processing.
- Evaluation of model performance on encrypted data.
- Comprehensive documentation and final project report.

## Prerequisites

- Python 3.x
- Required Python packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

To install the required Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage

### Processing Encrypted MRI Images

To process MRI images using Fully Homomorphic Encryption:

```bash
python EncryptedProcessingMedical.py
```

This script will:
- Encrypt the MRI image data.
- Perform classification tasks on the encrypted data.
- Output the results and performance metrics.

### Processing Plain MRI Images

To process MRI images without encryption:

```bash
python PlainProcessingMedical.py
```

This script will:
- Load the plaintext MRI image data.
- Perform classification tasks on the data.
- Output the results and performance metrics.

## Project Structure

```
├── EncryptedProcessing.py             # Script for processing encrypted data
├── EncryptedProcessingMedical.py      # Script for processing encrypted MRI images
├── PlainProcessing.py                 # Script for processing plaintext data
├── PlainProcessingMedical.py          # Script for processing plaintext MRI images
├── Trustworthy Machine Learning - Final Project Report - Justin Morera and Sebastian Perez.pdf
├── README.md                          # Project documentation
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
