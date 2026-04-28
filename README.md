# DS 4320 Project 2: Predicting Pitcher xERA Using MLB Statcast Pitch-Quality Metrics

## Executive Summary

This project examines whether MLB Statcast pitcher-season metrics can be used to predict season-level pitcher performance. Using Statcast-derived data from the 2018 through 2021 MLB seasons, I created a MongoDB document database where each modeling document represents one pitcher-season observation.

The final modeling dataset includes pitcher identifiers, season, expected ERA, average estimated wOBA allowed, plate appearances, fastball velocity, and fastball spin rate. The completed pipeline queries the MongoDB collection into Python, converts the documents into a pandas dataframe, trains machine learning models, and evaluates how well these Statcast-based features predict season-level xERA.

This project demonstrates how a document database can support sports analytics by storing structured pitcher-season records while still allowing the data to be queried, modeled, analyzed, and visualized in Python.

## Name

Dev Aswani

## NetID

vzu3vu

## DOI

[![DOI](https://zenodo.org/badge/1220257253.svg)](https://doi.org/10.5281/zenodo.19826893)

## Press Release

[Press Release](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/d41538bbe42e64388d993247867ce2e1ed5402d3/press_release.md)

## Data

[Background Reading and Data Folder](https://myuva-my.sharepoint.com/:f:/g/personal/vzu3vu_virginia_edu/IgAgeeFylog8T62lp4NTkhNDAVaDEMGznPdSu2cUngmeDrQ?e=M5qECC)

## Pipeline

[Data Preparation Notebook](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/879d42d0afbc4bb2a5473637adee84bca0b86917/data/project2_data_prep.ipynb)  
[Analysis Pipeline Notebook](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/0aecd90ef4f2b777f54aac2badf91fef9a975f03/pipeline/project2_pipeline.ipynb)  
[Analysis Pipeline Markdown](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/7024d1749249f52e1a6b222f1bd77451ece2aa59/pipeline/project2_pipeline.md)

## License

[MIT License](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/85c8d3d572dc5cebf8b2b804db764a745765ad20/LICENSE)

---

## Problem Definition

### General Problem

Projecting athletic performance.

### Specific Problem

How well can pitcher-season Statcast pitch-quality metrics from the 2018–2021 MLB seasons predict season-level xERA for qualified pitchers?

### Rationale for Refinement

The original problem of projecting athletic performance is very broad, so this project narrows the scope to Major League Baseball pitchers and focuses specifically on expected ERA as the outcome of interest. This refinement makes the problem measurable, data-driven, and appropriate for a document-model pipeline because Statcast-derived pitcher-season records can be stored as MongoDB documents and then queried for predictive modeling.

Pitching is a strong refinement because pitcher performance depends heavily on repeatable process-based skills such as velocity, spin rate, pitch movement, pitch type, command, and contact quality allowed. By focusing on qualified pitchers from the 2018 through 2021 seasons, the project creates a manageable but meaningful dataset that supports both document storage and predictive modeling.

### Motivation

Traditional pitching statistics such as ERA can be useful, but they do not always isolate a pitcher’s actual skill because they can be influenced by defense, ballpark conditions, sequencing, and random variation in batted-ball outcomes. Statcast makes it possible to evaluate pitchers using the physical characteristics of each pitch, including velocity and spin rate, along with what happens when hitters make contact.

MLB’s expected metrics are especially useful because they aim to credit the pitcher for the quality of the event at the moment of contact rather than for downstream factors like defense or weather. This makes pitcher evaluation a strong setting for a data science project because the domain is rich, the data is detailed, and the problem naturally supports a document database and predictive pipeline.

### Headline of Press Release

[Statcast Data Reveals What Truly Drives Pitching Performance](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/d41538bbe42e64388d993247867ce2e1ed5402d3/press_release.md)

---

## Domain Exposition

This project exists within the domain of sports analytics, specifically baseball pitching analysis. Modern baseball analysis relies heavily on Statcast, which tracks detailed pitch data such as velocity, spin rate, pitch type, release characteristics, and contact outcomes that can be summarized into pitcher-season performance metrics. These metrics allow analysts to evaluate pitchers based on the quality of their pitches rather than only on traditional results.

By combining large-scale pitch-level data with statistical modeling, this domain enables more accurate predictions of player performance and supports data-driven decision-making in scouting, player development, and roster construction.

### Terminology

| Term | Definition | Why It Matters |
|---|---|---|
| Statcast | MLB tracking system that records pitch-level data | Source of the dataset |
| Velocity | Speed of a pitch in miles per hour | Impacts hitter reaction time and pitch effectiveness |
| Spin Rate | Rotation of the baseball measured in revolutions per minute | Affects pitch movement and deception |
| Pitch Type | Classification of the pitch thrown, such as fastball, slider, or curveball | Different pitch types produce different outcomes |
| xERA | Expected ERA based on quality of contact, strikeouts, and walks | Better reflects pitcher skill than traditional ERA |
| Strikeout Rate | Percentage of batters struck out | Measures pitcher dominance |
| Walk Rate | Percentage of batters walked | Measures pitcher control |
| Whiff Rate | Percentage of swings that miss | Indicates pitch effectiveness |
| Hard-Hit Rate | Percentage of batted balls hit with high exit velocity | Measures damaging contact allowed |
| Command | Ability to locate pitches accurately | Important factor in preventing strong contact |

### Background Reading

[Background readings folder](https://myuva-my.sharepoint.com/:f:/g/personal/vzu3vu_virginia_edu/IgAgeeFylog8T62lp4NTkhNDAVaDEMGznPdSu2cUngmeDrQ?e=IhukYt)

### Background Reading Table

| Title | Description | Link |
|---|---|---|
| MLB Statcast Glossary | Overview of Statcast system and metrics | [link](https://myuva-my.sharepoint.com/:u:/g/personal/vzu3vu_virginia_edu/IQAO7UCZIQo6SL5j26yVLIjPATQdThfdIIjx9S3Yy75EGdI?e=tlPlw5) |
| Expected ERA/xERA | Explains expected ERA and why it matters | [link](https://myuva-my.sharepoint.com/:u:/g/personal/vzu3vu_virginia_edu/IQD5xxOJIRiyR6RwzZjCTH7kAf8uLoJyru5xa8rTWl1j_Eo?e=mLVm9n) |
| Velocity | Defines pitch velocity and how it is measured | [link](https://myuva-my.sharepoint.com/:u:/g/personal/vzu3vu_virginia_edu/IQDxCW-rHSdzTbYIwV9Kc8ykAcg4C_Dk7Bk4XzMR5cCfBrE?e=y84mT1) |
| Spin Rate | Explains spin rate and its impact on pitch movement | [link](https://myuva-my.sharepoint.com/:u:/g/personal/vzu3vu_virginia_edu/IQAn6ao71Z-sRZ4ObFA4-8sBAZ8x80MiIcltz5gM80a-C84?e=AjImWy) |
| Pitch Modeling Primer | Explains how pitch data can be used in modeling | [link](https://myuva-my.sharepoint.com/:u:/g/personal/vzu3vu_virginia_edu/IQD5wOtRxXqAS43VNquZXX5dAdyFVvHXXFPuM9Rg6tfrsEY?e=7lWfhU) |

---

## Data Creation

### Data Acquisition

This project uses MLB Statcast-derived pitcher data from the 2018 through 2021 seasons. The raw data sources include pitcher expected statistics, pitcher pitch-arsenal speed, and pitcher pitch-arsenal spin. These sources are merged by pitcher and season to create a final pitcher-season modeling dataset.

Each observation in the final modeling dataset represents one pitcher in one MLB season. Therefore, each MongoDB document in the final modeling collection corresponds to a single pitcher-season observation rather than a single pitch. The final documents include pitcher identification, season, xERA, average estimated wOBA allowed, plate appearances, fastball velocity, and fastball spin rate.

After the raw data is loaded, the preparation process standardizes the relevant columns, merges the expected-statistics, velocity, and spin datasets, removes rows with missing values in critical modeling fields, and inserts the cleaned pitcher-season records into MongoDB.

### Data Processing Pipeline

The data pipeline transforms Statcast-derived pitcher data into a MongoDB document database and then into a modeling dataset. The main steps are:

1. **Raw Data Collection**  
   Pitcher expected-statistics, pitch-arsenal speed, and pitch-arsenal spin datasets are collected for the 2018 through 2021 MLB seasons.

2. **Feature Selection**  
   The project keeps pitcher-season variables needed for modeling, including pitcher ID, pitcher name, season, xERA, average estimated wOBA allowed, plate appearances, fastball velocity, and fastball spin rate.

3. **Cleaning**  
   Rows with missing values in critical fields are removed. Column names are standardized so that MongoDB documents follow a consistent soft schema.

4. **Merging**  
   The expected-statistics, velocity, and spin datasets are merged by pitcher and season to create a complete pitcher-season modeling table.

5. **Document Creation**  
   Each cleaned pitcher-season row is converted into a dictionary, allowing it to be inserted into MongoDB as an individual document.

6. **MongoDB Storage**  
   The cleaned pitcher-season records are stored in the `pitcher_season_model_data` MongoDB collection. This satisfies the document-model requirement and allows flexible querying.

7. **Modeling Table Construction**  
   The analysis pipeline queries MongoDB into a pandas dataframe and uses the pitcher-season documents as the modeling dataset for machine learning.

### Code Table

| File | Description | Link |
|---|---|---|
| `project2_data_prep.ipynb` | Loads raw Statcast data, keeps relevant pitcher variables, cleans missing values, standardizes column names, converts rows to MongoDB-ready documents, and inserts records into MongoDB | [notebook](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/879d42d0afbc4bb2a5473637adee84bca0b86917/data/project2_data_prep.ipynb) |
| `project2_pipeline.ipynb` | Queries MongoDB, converts documents into a dataframe, creates modeling features, trains a predictive model, evaluates results, and creates visualizations | [notebook](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/0aecd90ef4f2b777f54aac2badf91fef9a975f03/pipeline/project2_pipeline.ipynb) |
| `project2_pipeline.md` | Markdown export of the analysis pipeline notebook | [markdown](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/7024d1749249f52e1a6b222f1bd77451ece2aa59/pipeline/project2_pipeline.md) |

### Rationale for Critical Decisions

Pitcher-season data was chosen because the project’s target variable, xERA, is measured at the season level for each pitcher. Using pitcher-season documents keeps the unit of analysis consistent between the predictors and the outcome while still producing a dataset with well over 1,000 MongoDB documents. This also makes the project a natural fit for MongoDB because each document can store a complete pitcher-season record with identifying information, expected-performance metrics, workload, and fastball characteristics.

MongoDB was selected because the document model is well-suited for semi-structured sports tracking data. The document model is useful because the project combines multiple Statcast-derived sources, including expected statistics, pitch-arsenal velocity, and pitch-arsenal spin, into one flexible pitcher-season record. A document database can preserve this structure more naturally than a rigid table.

xERA was selected as the primary outcome because it better reflects pitcher skill than traditional ERA. ERA is affected by defense, sequencing, and randomness, while xERA attempts to measure expected run prevention based on strikeouts, walks, and quality of contact. This makes it a stronger target variable for a predictive model based on pitch-quality features.

### Bias Identification

Several sources of bias may be present in the data creation process. Selection bias can occur if the dataset includes only qualified pitchers, because injured players, relievers, and lower-volume pitchers may be excluded. Survivorship bias may also occur because pitchers who remain qualified across a season are more likely to be successful or healthy.

Measurement bias may exist because Statcast data relies on tracking technology that can produce small errors in variables such as velocity, spin rate, launch speed, and launch angle. Contextual bias may also arise because pitch outcomes depend on opposing hitters, defensive support, ballpark conditions, weather, and game situation, which are not fully captured by the selected features.

### Bias Mitigation

This project reduces bias by using multiple seasons of data from 2018 through 2021 rather than relying on a single year. The use of xERA also helps reduce the influence of defense and random variation because expected metrics focus more on the quality of the event than on the final result of the play.

The analysis uses pitcher-season summaries and plate appearances to reduce the influence of random single-event outcomes and to make the modeling unit consistent with the season-level xERA target. Exploratory analysis is used to identify missing values, outliers, and unusual observations. Filtering decisions are documented so that limitations are transparent and the results can be interpreted appropriately.

---

## Metadata

### Soft-Schema Guidelines

The primary MongoDB collection used for the analysis is `pitcher_season_model_data`. Each document represents one pitcher-season observation from the 2018 through 2021 MLB seasons. This means the unit of observation is not a single pitch, but a summarized season-level record for a specific pitcher in a specific year.

Documents follow a consistent soft schema with fields for pitcher identification, season, expected run-prevention performance, expected contact quality, workload, and fastball characteristics. Core identifying fields include `pitcher_id`, `pitcher_name`, and `season`. The target variable is `xERA`, while the main predictive features are `avg_estimated_woba`, `plate_appearances`, `fastball_velocity`, and `fastball_spin`.

Numerical variables are stored as integers or floats, while pitcher names are stored as strings. During data preparation, raw Statcast expected-stat and pitch-arsenal data are cleaned, merged by pitcher and season, converted into dictionaries, and uploaded to MongoDB as documents. Missing values in critical modeling fields are removed before analysis so that the final modeling collection is consistent and ready for querying.

### Data Summary

| Collection | Description | Unit of Observation | Document Count |
|---|---|---:|---:|
| `pitcher_season_model_data` | Final merged modeling dataset containing season-level pitcher Statcast features from 2018–2021 | One document per pitcher-season | 2,967 |
| `pitcher_expected_stats_raw` | Raw Statcast expected pitcher statistics by season | One document per pitcher-season expected-stat record | 3,274 |
| `pitcher_pitch_arsenal_speed_raw` | Raw Statcast pitch arsenal speed data by season | One document per pitcher-season pitch-arsenal speed record | 3,166 |
| `pitcher_pitch_arsenal_spin_raw` | Raw Statcast pitch arsenal spin data by season | One document per pitcher-season pitch-arsenal spin record | 3,166 |

The primary analysis uses the `pitcher_season_model_data` collection, which contains 2,967 pitcher-season observations across 1,271 unique pitchers and the 2018, 2019, 2020, and 2021 MLB seasons.

### Data Dictionary

| Feature | Type | Description | Example |
|---|---|---|---|
| `pitcher_id` | Integer | Unique MLB identifier for each pitcher | `572971` |
| `pitcher_name` | String | Pitcher's name in last-name, first-name format | `Keuchel, Dallas` |
| `season` | Integer | MLB season for the pitcher-season observation | `2018` |
| `xERA` | Float | Expected earned run average; the prediction target measuring expected run prevention | `3.60` |
| `avg_estimated_woba` | Float | Average estimated weighted on-base average allowed, based on quality of contact and expected outcomes | `0.293` |
| `plate_appearances` | Integer | Number of plate appearances included for that pitcher-season | `874` |
| `fastball_velocity` | Float | Average fastball velocity in miles per hour | `89.7` |
| `fastball_spin` | Float | Average fastball spin rate in revolutions per minute | `2166.0` |

### Quantification of Uncertainty

| Feature | Mean | Std. Dev. | Min | Max | Interpretation |
|---|---:|---:|---:|---:|---|
| `xERA` | 5.49 | 6.22 | 0.00 | 190.84 | xERA has a wide range because some pitcher-season observations contain extreme expected run-prevention values. These outliers should be interpreted carefully. |
| `fastball_velocity` | 92.98 | 2.95 | 64.00 | 101.00 | Most pitchers cluster around the low-to-mid 90s, but the minimum suggests some unusual or low-volume pitcher-season observations. |
| `fastball_spin` | 2255.47 | 162.00 | 1448.00 | 2891.00 | Spin rate varies meaningfully across pitchers and reflects differences in pitch movement and pitch quality. |
| `avg_estimated_woba` | Calculated in pipeline | Calculated in pipeline | Calculated in pipeline | Calculated in pipeline | This variable is conceptually related to xERA because both are expected-performance metrics, so it may introduce overlap between predictors and the target. |
| `plate_appearances` | Calculated in pipeline | Calculated in pipeline | Calculated in pipeline | Calculated in pipeline | Plate appearances measure workload and sample size. Low-volume pitchers may produce noisier season-level observations. |

Pitcher-season outcomes contain uncertainty because xERA, estimated wOBA, velocity, and spin rate are all summaries of many individual baseball events. Some uncertainty comes from measurement error in Statcast tracking technology, while additional uncertainty comes from differences in opponent quality, ballpark conditions, pitch mix, injuries, role changes, and sample size. The model reduces single-event randomness by using season-level summaries, but the results should still be interpreted as predictive rather than causal.

---
## Statistical Interpretation

The summary statistics indicate that most pitcher-seasons cluster within a relatively narrow range of xERA, fastball velocity, fastball spin, and expected contact quality. However, there is still meaningful variation across pitchers, which suggests that Statcast-based features can help explain differences in pitching performance.

The model results show that adding expected contact quality, especially average estimated wOBA, substantially improves predictive performance. Fastball velocity and fastball spin alone provide limited explanatory power, but when combined with expected outcome metrics, the model captures a much larger portion of variation in xERA.

The Random Forest model performs better than the baseline linear regression model, suggesting that the relationship between pitch-quality metrics and xERA is not purely linear. However, the results should be interpreted carefully because average estimated wOBA is conceptually related to xERA, meaning some predictive strength may come from overlapping information in the target and predictor variables.

Overall, the analysis suggests that pitcher performance is not random. Statcast pitch-quality metrics contain meaningful predictive signal, but xERA is still influenced by additional factors such as pitch mix, command, opponent quality, sequencing, and game context.
