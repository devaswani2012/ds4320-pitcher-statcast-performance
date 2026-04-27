# Project 2 Pipeline: Predicting Pitcher xERA with Statcast Pitch-Quality Metrics

This notebook implements the problem solution pipeline for DS 4320 Project 2. The pipeline connects to MongoDB, queries the pitcher-season Statcast collection, converts the documents into a pandas dataframe, cleans the modeling data, trains a machine learning model, evaluates performance, and creates a visualization comparing actual and predicted xERA.

The specific problem is:

**How well can pitcher-season Statcast pitch-quality metrics from the 2018–2021 MLB seasons predict season-level xERA for qualified pitchers?**


```python
!pip install pandas numpy matplotlib scikit-learn pymongo python-dotenv
```


```python
import os
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymongo import MongoClient
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

os.makedirs("logs", exist_ok=True)

log_file = "logs/project2_pipeline.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

logging.info("Pipeline started successfully.")

print(f"Log file created at: {log_file}")
```

    Log file created at: logs/project2_pipeline.log



```python
from pymongo import MongoClient

MONGO_URI = "removed for submission"

try:
    client = MongoClient(MONGO_URI)
    db = client["project_2"]
    collection = db["pitcher_season_model_data"]

    document_count = collection.count_documents({})
    print(f"Connected to MongoDB. Documents in collection: {document_count}")

    logging.info(f"Connected to MongoDB. Documents in collection: {document_count}")

except Exception as e:
    logging.error(f"MongoDB connection failed: {e}")
    raise
```

    Connected to MongoDB. Documents in collection: 2967


## 2. Query MongoDB into a DataFrame

The MongoDB collection stores one document per pitcher-season observation. This query loads the documents and converts them into a pandas dataframe for analysis.


```python
try:
    documents = list(collection.find({}))
    df = pd.DataFrame(documents)

    print("Initial dataframe shape:", df.shape)
    display(df.head())

    logging.info(f"Loaded dataframe with shape {df.shape}")

except Exception as e:
    logging.error(f"Failed to load MongoDB documents into dataframe: {e}")
    raise
```

    Initial dataframe shape: (2967, 9)




  <div id="df-d5d5b183-dac9-4b6c-945e-25634b9cc41d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>_id</th>
      <th>pitcher_id</th>
      <th>pitcher_name</th>
      <th>season</th>
      <th>xERA</th>
      <th>avg_estimated_woba</th>
      <th>plate_appearances</th>
      <th>fastball_velocity</th>
      <th>fastball_spin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>69ebc1c26aeae905745c854e</td>
      <td>572971</td>
      <td>Keuchel, Dallas</td>
      <td>2018</td>
      <td>3.60</td>
      <td>0.293</td>
      <td>874</td>
      <td>89.7</td>
      <td>2166.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>69ebc1c26aeae905745c854f</td>
      <td>448306</td>
      <td>Shields, James</td>
      <td>2018</td>
      <td>5.33</td>
      <td>0.350</td>
      <td>871</td>
      <td>89.5</td>
      <td>2271.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69ebc1c26aeae905745c8550</td>
      <td>453286</td>
      <td>Scherzer, Max</td>
      <td>2018</td>
      <td>2.56</td>
      <td>0.248</td>
      <td>866</td>
      <td>94.4</td>
      <td>2487.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>69ebc1c26aeae905745c8551</td>
      <td>607536</td>
      <td>Freeland, Kyle</td>
      <td>2018</td>
      <td>3.76</td>
      <td>0.299</td>
      <td>844</td>
      <td>91.8</td>
      <td>2267.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>69ebc1c26aeae905745c8552</td>
      <td>446372</td>
      <td>Kluber, Corey</td>
      <td>2018</td>
      <td>3.18</td>
      <td>0.276</td>
      <td>842</td>
      <td>92.0</td>
      <td>2410.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d5d5b183-dac9-4b6c-945e-25634b9cc41d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d5d5b183-dac9-4b6c-945e-25634b9cc41d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d5d5b183-dac9-4b6c-945e-25634b9cc41d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




```python
# Remove MongoDB object ID because it is not useful for modeling
if "_id" in df.columns:
    df = df.drop(columns=["_id"])

print("Columns in dataset:")
print(df.columns.tolist())
```

    Columns in dataset:
    ['pitcher_id', 'pitcher_name', 'season', 'xERA', 'avg_estimated_woba', 'plate_appearances', 'fastball_velocity', 'fastball_spin']


## 3. Data Cleaning

The model uses fastball velocity and fastball spin to predict xERA. Rows with missing values in these required fields are removed before modeling.


```python
required_columns = [
    "pitcher_id",
    "pitcher_name",
    "season",
    "xERA",
    "avg_estimated_woba",
    "plate_appearances",
    "fastball_velocity",
    "fastball_spin"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

model_df = df[required_columns].copy()

numeric_columns = ["season", "xERA", "fastball_velocity", "fastball_spin"]

for col in numeric_columns:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

before_drop = model_df.shape[0]
model_df = model_df.dropna(subset=["xERA", "fastball_velocity", "fastball_spin"])
after_drop = model_df.shape[0]

print(f"Rows before cleaning: {before_drop}")
print(f"Rows after cleaning: {after_drop}")
print(f"Rows dropped: {before_drop - after_drop}")

display(model_df.head())

logging.info(f"Rows before cleaning: {before_drop}")
logging.info(f"Rows after cleaning: {after_drop}")
```

    Rows before cleaning: 2967
    Rows after cleaning: 2967
    Rows dropped: 0




  <div id="df-bc6e99b7-5aa0-4279-beaf-6814f872fc8d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pitcher_id</th>
      <th>pitcher_name</th>
      <th>season</th>
      <th>xERA</th>
      <th>avg_estimated_woba</th>
      <th>plate_appearances</th>
      <th>fastball_velocity</th>
      <th>fastball_spin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>572971</td>
      <td>Keuchel, Dallas</td>
      <td>2018</td>
      <td>3.60</td>
      <td>0.293</td>
      <td>874</td>
      <td>89.7</td>
      <td>2166.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>448306</td>
      <td>Shields, James</td>
      <td>2018</td>
      <td>5.33</td>
      <td>0.350</td>
      <td>871</td>
      <td>89.5</td>
      <td>2271.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>453286</td>
      <td>Scherzer, Max</td>
      <td>2018</td>
      <td>2.56</td>
      <td>0.248</td>
      <td>866</td>
      <td>94.4</td>
      <td>2487.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>607536</td>
      <td>Freeland, Kyle</td>
      <td>2018</td>
      <td>3.76</td>
      <td>0.299</td>
      <td>844</td>
      <td>91.8</td>
      <td>2267.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>446372</td>
      <td>Kluber, Corey</td>
      <td>2018</td>
      <td>3.18</td>
      <td>0.276</td>
      <td>842</td>
      <td>92.0</td>
      <td>2410.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-bc6e99b7-5aa0-4279-beaf-6814f872fc8d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-bc6e99b7-5aa0-4279-beaf-6814f872fc8d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-bc6e99b7-5aa0-4279-beaf-6814f872fc8d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>



## 4. Exploratory Summary

This section summarizes the final modeling dataset and checks the distribution of the main variables.


```python
summary_stats = model_df[["xERA", "fastball_velocity", "fastball_spin"]].describe()

display(summary_stats)
```



  <div id="df-f151f5ee-c35d-4fb2-b230-b23f62991e0d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>xERA</th>
      <th>fastball_velocity</th>
      <th>fastball_spin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2967.000000</td>
      <td>2967.000000</td>
      <td>2967.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.492727</td>
      <td>92.975598</td>
      <td>2255.469161</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.217268</td>
      <td>2.953280</td>
      <td>162.000938</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>64.000000</td>
      <td>1448.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.680000</td>
      <td>91.500000</td>
      <td>2150.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.510000</td>
      <td>93.200000</td>
      <td>2257.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.770000</td>
      <td>94.700000</td>
      <td>2364.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>190.840000</td>
      <td>101.000000</td>
      <td>2891.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f151f5ee-c35d-4fb2-b230-b23f62991e0d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f151f5ee-c35d-4fb2-b230-b23f62991e0d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f151f5ee-c35d-4fb2-b230-b23f62991e0d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_d5e83a37-699a-4305-8313-47c6db407bb6">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('summary_stats')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_d5e83a37-699a-4305-8313-47c6db407bb6 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('summary_stats');
      }
      })();
    </script>
  </div>

    </div>
  </div>




```python
print("Number of pitcher-season observations:", len(model_df))
print("Number of unique pitchers:", model_df["pitcher_id"].nunique())
print("Seasons included:", sorted(model_df["season"].unique()))
```

    Number of pitcher-season observations: 2967
    Number of unique pitchers: 1271
    Seasons included: [np.int64(2018), np.int64(2019), np.int64(2020), np.int64(2021)]


## 5. Feature Selection

The prediction target is `xERA`. The explanatory variables are:

- `fastball_velocity`
- `fastball_spin`
- `season`

Velocity and spin rate are core Statcast pitch-quality variables. Season is included to account for year-to-year differences across the 2018–2021 period.


```python
features = [
    "avg_estimated_woba",
    "plate_appearances",
    "fastball_velocity",
    "fastball_spin"
]

target = "xERA"

X = model_df[features]
y = model_df[target]

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
```

    Feature matrix shape: (2967, 4)
    Target shape: (2967,)


## 6. Train/Test Split

The data is split into training and testing sets. The model is trained on 80% of the pitcher-season observations and evaluated on the remaining 20%.


```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])
```

    Training rows: 2373
    Testing rows: 594


## 7. Baseline Model: Linear Regression

Linear regression is used as a baseline model because it is simple, interpretable, and useful for understanding the direction of relationships between Statcast pitch-quality variables and xERA.


```python
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
linear_r2 = r2_score(y_test, linear_predictions)

print("Linear Regression Results")
print("MAE:", round(linear_mae, 4))
print("RMSE:", round(linear_rmse, 4))
print("R²:", round(linear_r2, 4))

logging.info("Linear regression model trained.")
logging.info(f"Linear MAE: {linear_mae}")
logging.info(f"Linear RMSE: {linear_rmse}")
logging.info(f"Linear R2: {linear_r2}")
```

    Linear Regression Results
    MAE: 1.4226
    RMSE: 2.0328
    R²: 0.6816


The updated model shows a substantial improvement in predictive performance compared to the baseline specification. The inclusion of avg_estimated_woba significantly increases the model’s explanatory power, resulting in an R² of approximately 0.68. This indicates that the model is able to explain a large portion of the variation in pitcher xERA.

The reduction in RMSE and MAE further suggests that predictions are meaningfully closer to actual values. This improvement highlights the importance of including contact-quality metrics when modeling pitcher performance. While velocity and spin rate provide useful information about pitch characteristics, they are not sufficient on their own to explain outcomes.

The results demonstrate that expected contact metrics, such as estimated wOBA, play a critical role in predicting run prevention, as they capture the quality of interactions between pitchers and hitters.


```python
coef_df = pd.DataFrame({
    "feature": features,
    "coefficient": linear_model.coef_
})

display(coef_df)
```



  <div id="df-53dfe824-be18-4499-bad6-480b466e46a6" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>avg_estimated_woba</td>
      <td>68.530549</td>
    </tr>
    <tr>
      <th>1</th>
      <td>plate_appearances</td>
      <td>0.000429</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fastball_velocity</td>
      <td>0.041271</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fastball_spin</td>
      <td>0.000774</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-53dfe824-be18-4499-bad6-480b466e46a6')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-53dfe824-be18-4499-bad6-480b466e46a6 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-53dfe824-be18-4499-bad6-480b466e46a6');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_cab97dec-07d9-4762-b410-5c4f07eb6ad2">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('coef_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_cab97dec-07d9-4762-b410-5c4f07eb6ad2 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('coef_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



## 8. Machine Learning Model: Random Forest Regressor

A Random Forest Regressor is used as the main machine learning model. This model can capture nonlinear relationships between pitch-quality metrics and xERA, which is useful because pitcher performance is unlikely to depend on velocity or spin in a perfectly linear way.


```python
rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=5,
    min_samples_leaf=5
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("Random Forest Results")
print("MAE:", round(rf_mae, 4))
print("RMSE:", round(rf_rmse, 4))
print("R²:", round(rf_r2, 4))

logging.info("Random forest model trained.")
logging.info(f"Random Forest MAE: {rf_mae}")
logging.info(f"Random Forest RMSE: {rf_rmse}")
logging.info(f"Random Forest R2: {rf_r2}")
```

    Random Forest Results
    MAE: 0.1773
    RMSE: 1.4132
    R²: 0.8461


## 9. Model Comparison

The baseline linear regression model is compared with the Random Forest model using MAE, RMSE, and R².


```python
results_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [linear_mae, rf_mae],
    "RMSE": [linear_rmse, rf_rmse],
    "R2": [linear_r2, rf_r2]
})

display(results_df)
```



  <div id="df-11f7d172-1591-46c0-9ded-b3657ceb284e" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>R2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>1.422563</td>
      <td>2.032754</td>
      <td>0.681632</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest</td>
      <td>0.177344</td>
      <td>1.413220</td>
      <td>0.846121</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-11f7d172-1591-46c0-9ded-b3657ceb284e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-11f7d172-1591-46c0-9ded-b3657ceb284e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-11f7d172-1591-46c0-9ded-b3657ceb284e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_a4931442-1f74-4a8d-9382-bdf7a40032dc">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('results_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_a4931442-1f74-4a8d-9382-bdf7a40032dc button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('results_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



The Random Forest model substantially outperforms linear regression, suggesting that the relationship between Statcast pitch-quality metrics and xERA is nonlinear. Tree-based methods are able to capture interactions and threshold effects that linear models cannot.

While the Random Forest model achieves very strong predictive performance, it is important to note that avg_estimated_woba is conceptually related to xERA, as both are expected performance metrics derived from similar underlying inputs.

Because of this, the model may partially rely on information that is already embedded in the target variable. This can inflate predictive accuracy and should be interpreted as a limitation rather than purely as model strength.

However, this result still demonstrates that expected contact-quality metrics are highly informative predictors of pitcher performance.

## 10. Feature Importance

Feature importance is used to identify which Statcast variables contribute most to the Random Forest model’s predictions.


```python
importance_df = pd.DataFrame({
    "feature": features,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

display(importance_df)
```



  <div id="df-107bd73c-f0b3-4354-9b2a-dce25e2c5c57" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>avg_estimated_woba</td>
      <td>0.999974</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fastball_spin</td>
      <td>0.000014</td>
    </tr>
    <tr>
      <th>1</th>
      <td>plate_appearances</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fastball_velocity</td>
      <td>0.000002</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-107bd73c-f0b3-4354-9b2a-dce25e2c5c57')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-107bd73c-f0b3-4354-9b2a-dce25e2c5c57 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-107bd73c-f0b3-4354-9b2a-dce25e2c5c57');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_ea2f1cb6-310d-41fc-83f6-4d3928cc1255">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('importance_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_ea2f1cb6-310d-41fc-83f6-4d3928cc1255 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('importance_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>




```python
plt.figure(figsize=(8, 5))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance for Predicting xERA")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```


    
![png](project2_pipeline_files/project2_pipeline_28_0.png)
    


## 11. Visualization: Actual vs. Predicted xERA

This visualization compares actual xERA values with predicted xERA values from the Random Forest model. A stronger model should produce points that fall closer to the diagonal reference line.


```python
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 7))

plt.scatter(y_test, rf_predictions, alpha=0.7)

plt.plot([0, 10], [0, 10], linestyle="--")

plt.xlim(0, 10)
plt.ylim(0, 10)

ticks = np.arange(0, 10.1, 2)
plt.xticks(ticks)
plt.yticks(ticks)

plt.xlabel("Actual xERA")
plt.ylabel("Predicted xERA")
plt.title("Actual vs Predicted xERA")

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```


    
![png](project2_pipeline_files/project2_pipeline_30_0.png)
    



```python
prediction_results = pd.DataFrame({
    "actual_xERA": y_test.values,
    "predicted_xERA": rf_predictions
})

prediction_results["residual"] = prediction_results["actual_xERA"] - prediction_results["predicted_xERA"]

display(prediction_results.head(10))
```



  <div id="df-21fe881c-ff27-4871-9edc-41141e5db963" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actual_xERA</th>
      <th>predicted_xERA</th>
      <th>residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.88</td>
      <td>2.866832</td>
      <td>0.013168</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.80</td>
      <td>11.311310</td>
      <td>0.488690</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37.28</td>
      <td>44.085161</td>
      <td>-6.805161</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.15</td>
      <td>3.302089</td>
      <td>-0.152089</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.47</td>
      <td>3.438081</td>
      <td>0.031919</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.42</td>
      <td>3.433340</td>
      <td>-0.013340</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.68</td>
      <td>3.680063</td>
      <td>-0.000063</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.99</td>
      <td>3.113209</td>
      <td>-0.123209</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.24</td>
      <td>4.380253</td>
      <td>-0.140253</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.88</td>
      <td>2.866832</td>
      <td>0.013168</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-21fe881c-ff27-4871-9edc-41141e5db963')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-21fe881c-ff27-4871-9edc-41141e5db963 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-21fe881c-ff27-4871-9edc-41141e5db963');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>



## 12. Residual Analysis

Residuals measure the difference between actual xERA and predicted xERA. Smaller residuals indicate more accurate predictions.


```python
plt.figure(figsize=(8, 5))

plt.scatter(
    prediction_results["predicted_xERA"],
    prediction_results["residual"],
    alpha=0.7
)

plt.axhline(0, linestyle="--")

plt.xlim(0, 10)
plt.ylim(-10, 10)

plt.xticks(np.arange(0, 10.1, 2))
plt.yticks(np.arange(-10, 10.1, 2))

plt.xlabel("Predicted xERA")
plt.ylabel("Residual")
plt.title("Residual Plot for Random Forest xERA Predictions")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```


    
![png](project2_pipeline_files/project2_pipeline_33_0.png)
    


## 13. Analysis Rationale

The analysis begins by querying the MongoDB collection and converting the documents into a dataframe. This step demonstrates that the project uses MongoDB as the primary storage layer rather than relying only on local CSV files.

The model uses fastball velocity, fastball spin rate, and season as predictors of xERA. Velocity and spin rate are included because they are core pitch-quality metrics in Statcast analysis. Season is included to account for possible changes across the 2018–2021 period.

Linear regression is used as a baseline because it is interpretable and provides a simple comparison point. Random Forest is used as the main machine learning model because it can capture nonlinear relationships and interactions between predictors.

The model is evaluated using MAE, RMSE, and R². MAE and RMSE measure the size of prediction errors, while R² measures the share of variation in xERA explained by the model.

## 14. Visualization Rationale

The actual-versus-predicted xERA scatterplot is used because it directly shows whether the model is solving the prediction problem. If predicted values are close to actual values, the points should cluster near the diagonal reference line.

The residual plot is included to check whether prediction errors are randomly distributed or whether the model systematically overpredicts or underpredicts certain pitchers. These visualizations help evaluate both model accuracy and model limitations.

## 15. Conclusion

This pipeline shows how a MongoDB document database can support a predictive sports analytics workflow. The project begins with Statcast pitcher-season documents, queries them into Python, trains machine learning models, and visualizes prediction performance.

The results indicate whether Statcast pitch-quality metrics such as fastball velocity and spin rate contain useful information for predicting xERA. While the model may not explain all variation in pitcher performance, it provides a structured and reproducible way to connect pitch-quality data with expected run prevention.


```python
logging.info("Pipeline completed successfully.")
print("Pipeline completed successfully.")
```

    Pipeline completed successfully.

