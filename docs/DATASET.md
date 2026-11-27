# Dataset Documentation

## Dataset
NIH Malaria Cell Image Dataset

## Source
National Institutes of Health (NIH), United States.

## Description
The dataset contains microscopic cell images categorised into two classes:

- **Parasitized:** Red blood cell images containing malaria parasites.
- **Uninfected:** Red blood cell images without malaria parasites.

## Project Application
The dataset was used to train and evaluate a convolutional neural network (CNN) for binary malaria cell classification.

## Model Input
- Image dimensions: 64 × 64 pixels
- Number of channels: 3 (RGB)
- Input format: INT8 quantized for deployment

## Data Handling
The original dataset is not redistributed in this repository. Researchers should obtain it from its original source and follow the dataset's applicable terms of use.

## Reproducibility
The model training, validation, testing and quantization workflow is documented in the accompanying Google Colab notebook.
