# Exploratory Data Analysis on a Used Cars Dataset

Unlox AIML - Minor Project 1
**Student:** Chinmaya SD (USN 1AM25CI033)

A complete EDA (no machine learning) on 4,340 used-car listings: data inspection, quality checks, cleaning,
descriptive statistics, 8 charts and 7 evidence-based findings.

## Dataset
CarDekho used-car listings, originally published on Kaggle ("Car Data Datasets").
Copy used: `Car_dataset.csv` from https://github.com/PiyushBadhe/Car-Price-Prediction-ML

The dataset file is not included in this repository. The script downloads it automatically if
`Car_dataset.csv` is not in the same folder.

## How to run
```bash
pip install -r requirements.txt
python ChinmayaSD_Cars_EDA_Project.py
```
All results are printed to the console and the charts are saved in the `charts/` folder.

## What the code does
1. Loads the data and inspects shape, columns, data types
2. Checks missing values, duplicates, inconsistent labels
3. Cleans the data (4,340 rows -> 3,575 rows: 763 duplicates and 2 impossible mileage values removed)
4. Descriptive statistics for numerical and categorical columns
5. Visual EDA: bar charts, histogram, scatter plots, box plot, correlation heatmap
6. Prints the key findings and limitations

## Charts
| File | Chart |
|---|---|
| charts/chart1_top_brands.png | Ten most common brands |
| charts/chart2_fuel_type.png | Listings by fuel type |
| charts/chart3_price_hist.png | Price distribution |
| charts/chart4_age_vs_price.png | Car age vs price |
| charts/chart5_km_vs_price.png | Kilometres driven vs price |
| charts/chart6_transmission_box.png | Price by transmission |
| charts/chart7_brand_avg_price.png | Average price by brand |
| charts/chart8_correlation_heatmap.png | Correlation heatmap |

The full written report is submitted separately as a PDF.
