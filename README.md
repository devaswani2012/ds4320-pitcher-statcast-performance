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

This project uses MLB Statcast pitcher data from the 2018 through 2021 seasons. The data was collected in Python using the `pybaseball` package, specifically the `statcast_pitcher_expected_stats` function for expected pitcher performance metrics and the `statcast_pitcher_pitch_arsenal` function for pitch-arsenal velocity and spin information. Instead of storing each individual pitch as the final modeling unit, the main dataset is organized at the pitcher-season level. Each document in the main modeling collection represents one pitcher in one MLB season.

The raw data acquisition process began by pulling three types of Statcast information for each season: expected pitcher statistics, average pitch speed data, and average pitch spin data. These raw tables were kept as separate MongoDB collections to preserve the original source structure. A cleaned modeling dataset was then created by selecting the relevant fields, renaming columns into a consistent format, merging the expected-statistics table with fastball velocity and fastball spin data, converting numeric columns into appropriate numeric types, removing rows with missing values in critical modeling fields, and dropping duplicate pitcher-season observations. The final modeling collection, `pitcher_season_model_data`, contains 2,967 pitcher-season documents.

### Data Processing Pipeline

The data pipeline transforms raw Statcast pitcher data into a document database and then into a modeling dataset. The main steps are:

1. **Data Collection**  
   Statcast expected pitcher statistics, pitch-arsenal average speed data, and pitch-arsenal average spin data are collected for each season from 2018 through 2021.

2. **Feature Selection**  
   The expected-statistics data is reduced to pitcher identifiers, pitcher names, season, xERA, average estimated wOBA, and plate appearances. The pitch-arsenal tables are reduced to pitcher identifiers, average fastball velocity, and average fastball spin.

3. **Cleaning**  
   Columns are renamed into a consistent schema. Numeric fields are converted into numeric data types using coercion to handle invalid values. Rows missing critical fields such as `xERA`, `fastball_velocity`, or `fastball_spin` are removed.

4. **Merging**  
   The expected-statistics data, velocity data, and spin data are merged by `pitcher_id`. This creates one modeling record per pitcher-season.

5. **Document Creation**  
   Each cleaned pitcher-season row is converted into a Python dictionary so it can be inserted into MongoDB as a document.

6. **MongoDB Storage**  
   The cleaned pitcher-season modeling documents are stored in the `pitcher_season_model_data` collection. The raw supporting tables are also stored in MongoDB as separate collections so the source data remains available for inspection.

7. **Modeling Table Construction**  
   The analysis pipeline queries the `pitcher_season_model_data` collection from MongoDB, converts the documents into a pandas dataframe, and uses the resulting dataframe for machine learning analysis and visualization.

### Code Table

| File | Description | Link |
|---|---|---|
| `project2_data_prep.ipynb` | Loads raw Statcast data, keeps relevant pitcher variables, cleans missing values, standardizes column names, converts rows to MongoDB-ready documents, and inserts records into MongoDB | [notebook](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/879d42d0afbc4bb2a5473637adee84bca0b86917/data/project2_data_prep.ipynb) |
| `project2_pipeline.ipynb` | Queries MongoDB, converts documents into a dataframe, creates modeling features, trains a predictive model, evaluates results, and creates visualizations | [notebook](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/0aecd90ef4f2b777f54aac2badf91fef9a975f03/pipeline/project2_pipeline.ipynb) |
| `project2_pipeline.md` | Markdown export of the analysis pipeline notebook | [markdown](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/7024d1749249f52e1a6b222f1bd77451ece2aa59/pipeline/project2_pipeline.md) |

### Rationale for Critical Decisions

The main unit of observation was set at the pitcher-season level because the project’s specific problem is to predict season-level xERA. Since xERA is a season-level pitcher performance metric, a pitcher-season document is the most appropriate modeling unit. This avoids incorrectly treating individual pitches as independent observations for a season-level outcome.

MongoDB was selected because the project still uses a document model to store both the cleaned modeling dataset and the raw supporting Statcast tables. The document structure makes it possible to preserve multiple related data sources while also producing a clean modeling collection for analysis. Storing the raw expected-statistics, velocity, and spin data in separate collections also helps with transparency because the modeling data can be traced back to its source tables.

Fastball velocity and fastball spin were selected because they are core Statcast pitch-quality metrics that describe the physical characteristics of a pitcher’s arsenal. Average estimated wOBA was included because it captures the expected quality of contact and plate-appearance outcomes allowed by the pitcher. Plate appearances were included as a volume measure because pitcher-season estimates are generally more reliable when they are based on larger samples.

Several judgment calls may introduce uncertainty. First, the project focuses on fastball velocity and fastball spin rather than every pitch type, which simplifies the model but may leave out important information about pitchers who rely heavily on breaking balls or off-speed pitches. Second, rows missing xERA, fastball velocity, or fastball spin are removed, which improves data consistency but may exclude pitchers with incomplete Statcast records. Third, average estimated wOBA is closely related to xERA, so the model’s predictive strength should be interpreted carefully because some predictor information overlaps conceptually with the target variable.

### Bias Identification

Several forms of bias could be introduced during data collection and preparation. Selection bias may occur because the dataset only includes pitchers who appear in Statcast pitcher tables and have the required expected-statistics, velocity, and spin information. Pitchers with very limited playing time, incomplete tracking data, or missing fastball information may be excluded. This means the final modeling dataset may overrepresent pitchers with larger workloads or more complete Statcast records.

Survivorship bias may also be present because pitchers who accumulate enough plate appearances to appear in the data are more likely to be healthy, active, and trusted by MLB teams. Measurement bias is also possible because Statcast variables such as velocity, spin rate, and expected outcome metrics are generated by tracking systems and modeling assumptions. Finally, omitted-variable bias may occur because the model does not fully account for defense, ballpark effects, opposing hitter quality, pitch location, pitch mix, injuries, or game situation.

### Bias Mitigation

This project reduces bias by using four seasons of data rather than relying on one season. Using multiple years helps reduce the influence of unusual single-season conditions and creates a larger sample of pitcher-seasons. The project also keeps the data creation process transparent by documenting which Statcast sources were used, which fields were retained, and which collections were uploaded to MongoDB.

Bias is also addressed by using xERA instead of traditional ERA as the target variable. xERA is better suited for pitcher evaluation because it focuses more on expected outcomes and quality of contact rather than only on final runs allowed, which can be affected by defense and luck. However, the project still acknowledges that xERA is not a perfect measure of pitcher skill. The final interpretation of the model should therefore treat the results as evidence of predictive signal in Statcast metrics, not as a complete explanation of pitcher performance.

---

## Metadata

### Soft-Schema Guidelines

The main MongoDB modeling collection is `pitcher_season_model_data`. Each document in this collection represents one pitcher-season observation. The document schema is intentionally simple and consistent because the collection is designed for machine learning analysis. Each document contains pitcher identification fields, a season field, the target variable, and several numeric predictor variables.

A typical document follows this structure:

```json
{
  "pitcher_id": 572971,
  "pitcher_name": "Keuchel, Dallas",
  "season": 2018,
  "xERA": 3.60,
  "avg_estimated_woba": 0.293,
  "plate_appearances": 874,
  "fastball_velocity": 89.7,
  "fastball_spin": 2166.0
}

The soft schema allows MongoDB to store the pitcher-season records as flexible documents while maintaining a consistent structure for analysis. Integer fields include identifiers, seasons, and plate appearance counts. Float fields include xERA, average estimated wOBA, fastball velocity, and fastball spin. String fields are used for pitcher names. Missing or invalid values in critical modeling fields are handled during data preparation before documents are inserted into the final modeling collection.
The project also stores raw supporting collections in MongoDB. These collections preserve the original expected-statistics, pitch-arsenal speed, and pitch-arsenal spin data before they are merged into the final modeling collection.
### Data Summary

| Collection | Description | Unit of Observation | Document Count |
|---|---|---:|---:|
| `pitcher_season_model_data` | Cleaned modeling dataset used in the analysis pipeline | One pitcher-season | 2,967 |
| `pitcher_expected_stats_raw` | Raw Statcast expected pitcher statistics by season | One pitcher-season expected-stat row | 3,274 |
| `pitcher_pitch_arsenal_speed_raw` | Raw Statcast pitch-arsenal average speed data | One pitcher-season arsenal-speed row | 3,166 |
| `pitcher_pitch_arsenal_spin_raw` | Raw Statcast pitch-arsenal average spin data | One pitcher-season arsenal-spin row | 3,166 |
| **Total** | All MongoDB documents uploaded across project collections | Mixed supporting and modeling documents | **12,573** |

### Data Dictionary

| Feature | Type | Description | Example |
|---|---|---|---|
| `_id` | ObjectId | MongoDB-generated unique document identifier. This is removed before modeling because it has no predictive meaning. | `69ebc1c26aeae905745c854e` |
| `pitcher_id` | Integer | Unique MLB identifier for the pitcher. Used to merge Statcast tables and distinguish pitchers with similar names. | `572971` |
| `pitcher_name` | String | Pitcher name as listed in the Statcast data. | `Keuchel, Dallas` |
| `season` | Integer | MLB season for the observation. | `2018` |
| `xERA` | Float | Expected earned run average. This is the target variable the model attempts to predict. | `3.60` |
| `avg_estimated_woba` | Float | Average estimated weighted on-base average allowed by the pitcher. Lower values generally indicate better expected pitcher outcomes. | `0.293` |
| `plate_appearances` | Integer | Number of plate appearances represented in the pitcher-season expected-statistics record. | `874` |
| `fastball_velocity` | Float | Average fastball velocity for the pitcher-season, measured in miles per hour. | `89.7` |
| `fastball_spin` | Float | Average fastball spin rate for the pitcher-season, measured in revolutions per minute. | `2166.0` |

### Quantification of Uncertainty

| Feature | Mean | Std. Dev. | Min | Max | Interpretation |
|---|---:|---:|---:|---:|---|
| `xERA` | 5.49 | 6.22 | 0.00 | 190.84 | The high maximum suggests extreme outliers may exist, likely from very small samples or unusual Statcast records. This creates uncertainty in model evaluation. |
| `fastball_velocity` | 92.98 | 2.95 | 64.00 | 101.00 | Most pitcher-seasons cluster near MLB fastball norms, but the minimum suggests possible low-volume pitchers, classification issues, or unusual observations. |
| `fastball_spin` | 2255.47 | 162.00 | 1448.00 | 2891.00 | Spin rate varies meaningfully across pitchers, but extreme low or high values may reflect pitch classification differences or small-sample uncertainty. |
| `avg_estimated_woba` | Not displayed in summary output | Not displayed in summary output | Not displayed in summary output | Not displayed in summary output | This variable is conceptually close to xERA because both relate to expected pitcher outcomes. Its use improves prediction but introduces interpretive uncertainty because it may overlap with the target variable. |
| `plate_appearances` | Not displayed in summary output | Not displayed in summary output | Not displayed in summary output | Not displayed in summary output | Plate appearances measure sample size. Pitcher-seasons with fewer plate appearances may produce less stable xERA and estimated wOBA values. |

Pitcher-season outcomes contain uncertainty because xERA, estimated wOBA, velocity, and spin rate are all summaries of many individual baseball events. Some uncertainty comes from measurement error in Statcast tracking technology, while additional uncertainty comes from differences in opponent quality, ballpark conditions, pitch mix, injuries, role changes, and sample size. The model reduces single-event randomness by using season-level summaries, but the results should still be interpreted as predictive rather than causal.

---
## Statistical Interpretation

The summary statistics indicate that most pitcher-seasons cluster within a relatively narrow range of xERA, fastball velocity, fastball spin, and expected contact quality. However, there is still meaningful variation across pitchers, which suggests that Statcast-based features can help explain differences in pitching performance.

The model results show that adding expected contact quality, especially average estimated wOBA, substantially improves predictive performance. Fastball velocity and fastball spin alone provide limited explanatory power, but when combined with expected outcome metrics, the model captures a much larger portion of variation in xERA.

The Random Forest model performs better than the baseline linear regression model, suggesting that the relationship between pitch-quality metrics and xERA is not purely linear. However, the results should be interpreted carefully because average estimated wOBA is conceptually related to xERA, meaning some predictive strength may come from overlapping information in the target and predictor variables.

Overall, the analysis suggests that pitcher performance is not random. Statcast pitch-quality metrics contain meaningful predictive signal, but xERA is still influenced by additional factors such as pitch mix, command, opponent quality, sequencing, and game context.
