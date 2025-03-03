# DataLit-InsideAirbnb

This repository contains data and code for analyzing Airbnb listings data, leveraging the [Inside Airbnb](http://insideairbnb.com/) dataset. The project is part of a Data Literacy or data science exercise, focusing on cleaning, exploring, modeling, and visualizing Airbnb data.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data](#data)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

The **DataLit-InsideAirbnb** project aims to:
1. **Collect** Airbnb data from [Inside Airbnb](http://insideairbnb.com/).
2. **Clean** and preprocess the data for analysis.
3. **Explore** and visualize key insights about Airbnb listings, hosts, and reviews.
4. **Model** relationships (e.g., predicting prices).
5. **Evaluate** model performance and compare different modeling approaches.



---

## Repository Structure

```
DataLit-InsideAirbnb
│
├── models/
│   ├── base_regression.py/          # Models for price prediction
│   ├── TabPFN.py                    
│   └── ...
│
├── preprocessing/
│   ├── preprocessing.py             # Script for data aggregation, cleaning, imputation etc.             
│   └── ...
│
├── results/
│   ├── Model_benchmark/
│   │   └── model_comparison_fixed_r.csv
│   └── ...
│
├── visualization/
│   ├── plot_model_bench.py
│   ├── price_difference_maps.ipynb  # Geospatial visualization of prediction errors
│   └── ...
│
├── environment.yml        # Conda environment (optional)
├── requirements.txt       # Python dependencies (if using pip)
├── README.md              # This README
└── ...
```

- **models**: All used models for prediction and code for model benchmarking.
- **results/**: Outputs such as figures, metrics, model comparison files, etc.
- **source/**: ???
- **environment.yml / requirements.txt**: Environment or package requirements.

---

## Getting Started
For a full demo with our dataset, our models, and the model benchmark, see our kaggle notebook: [https://www.kaggle.com/datasets/georgtirpitz/datalit-dataset](https://www.kaggle.com/code/georgtirpitz/datalit-models/).
### Prerequisites

- Python 3.8+ (or a version specified in `environment.yml` / `requirements.txt`)
- Common data science libraries:  
  - `pandas`  
  - `numpy`  
  - `matplotlib`  
  - `seaborn`  
  - `scikit-learn` (for modeling)  
  - …and others as listed in your environment file.

### Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/GeorgTirp/DataLit-InsideAirbnb.git
   cd DataLit-InsideAirbnb
   ```

2. **Install dependencies**:

   - Using `conda` (if you have an `environment.yml`):
     ```bash
     conda env create -f environment.yml
     conda activate datalit_airbnb
     ```
   - Or using `pip` (if you have a `requirements.txt`):
     ```bash
     pip install -r requirements.txt
     ```

### Data

- **Inside Airbnb data** The fully preprocessed data also be found on kaggle as uploaded dataset: https://www.kaggle.com/datasets/georgtirpitz/datalit-dataset
- Run any data-cleaning or preprocessing scripts (see `scripts/data_cleaning.py` or the relevant notebook) to generate the preprocessed data.

---

## Usage

Below are some common workflows:

1. **Data Cleaning**:  
   - Aggregation of available data sources useful for prediction (`listings.csv` & `reviews.csv`)
   - Fast image downloading using asynchronous libraries (`asyncio` & `aiohttp`)
   - Data cleaning, imputation, currency conversion etc.
   - Feature embeddings for structured data such as images & text (for tabular regression models)
   - Flexible and easy to use data read-in & saving methods for further preprocessing

2. **Exploratory Analysis**:  
   - Open the Kaggle notebook to copy and edit to use GPU resources as some of the models can be intractable for CPU.
   - Execute cells to run models and explore the dataset.

3. **Visualization**:  
   - Look at the generated plots in `results/` or in the notebooks.
   - The folder `visualization/` contains scripts that produce advanced plots (e.g. prediction error maps).

---

## Results

- **Model Performance**:  
  - `results/Model_benchmark/model_comparison.csv` compares R² scores or other metrics across different models.  
  - Figures (e.g., `.png` or `.pdf`) for model comparison are stored in `results/`.

- **Analysis Findings**:  
  - Any discussion of the findings and our methods can be found in the written report.

---

## Contributing

Contributions, suggestions, and improvements are welcome!  
1. **Fork** this repository.  
2. **Create** a new branch for your feature or bugfix.  
3. **Submit** a pull request describing your changes.

---

## License

Unless otherwise specified, this project is licensed under the [MIT License](LICENSE). See [LICENSE](LICENSE) for details.

*(If you use Inside Airbnb data, be mindful of any relevant data usage agreements or licenses.)*

---

## Contact

For questions, comments, or collaboration, you can reach us at:
- **Name**: Georg Tirpitz  
- **GitHub**: [@GeorgTirp](https://github.com/GeorgTirp)
- **Name**: Nils Klute  
- **GitHub**: [@NilsKlute](https://github.com/NilsKlute)
- **Name**: Mathis Nommensen
- **Github**: [@mathisnommensen](https://github.com/mathisnommensen)

Feel free to open an [issue](https://github.com/GeorgTirp/DataLit-InsideAirbnb/issues) on GitHub if you encounter any problems or have feature requests.

---

*Happy analyzing!*
