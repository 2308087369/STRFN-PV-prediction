# Solar Power Prediction Project

This repository contains the open-source materials related to our academic paper on solar power prediction. The main technologies, including the photovoltaic power prediction model and dataset processing methods, will be open-sourced upon the paper's publication.

## Dataset

The processed dataset we collected has been uploaded to Kaggle. You are welcome to download it from [https://www.kaggle.com/datasets/chrisstarsky/solar-power-dataset-from-dka-solar-center].
Or you can download the datasets by python:
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("chrisstarsky/solar-power-dataset-from-dka-solar-center")

print("Path to dataset files:", path)
```

## Solar Power Prediction Platform

We have also developed a solar power prediction platform that allows for one-click dataset processing and includes multiple pre-trained neural network models, such as STRFN. This platform is currently under development and in the commercialization phase, and therefore, it is not open-source at this time. You can see a demo of the platform in the following GIFs:

- `source\GIF 2025-7-10 16-54-40.gif`
![Demo 1](source/GIF%202025-7-10%2016-54-40.gif)
- `source\GIF 2025-7-10 17-09-15.gif`
![Demo 2](source/GIF%202025-7-10%2017-09-15.gif)
