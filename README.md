# DS 4320 Project 2: Predicting Pitcher xERA Using MLB Statcast Pitch-Quality Metrics

## Executive Summary

This project examines whether MLB Statcast pitch-quality metrics can be used to predict season-level pitcher performance. Using pitch-level Statcast data from the 2018 through 2021 MLB seasons, I created a MongoDB document database where each document represents an individual pitch. The dataset includes pitcher identifiers, pitch characteristics, game context, and outcome-related variables such as velocity, spin rate, pitch type, launch speed, launch angle, and estimated contact-quality metrics.

The completed pipeline queries the MongoDB collection into Python, converts the documents into a pandas dataframe, aggregates pitch-level information into pitcher-season features, and applies a machine learning model to predict season-level xERA. This project demonstrates how a document database can support large-scale sports analytics by preserving pitch-level detail while still allowing the data to be transformed into a modeling table for analysis and visualization.

## Name

Dev Aswani

## NetID

vzu3vu

## DOI

[INSERT DOI BADGE OR DOI LINK]

## Press Release

[Press Release](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/84ec35330b6c09a9f7b75b347b165d09c304475a/press_release.md)

## Data

[Background Reading and Data Folder](https://myuva-my.sharepoint.com/:f:/g/personal/vzu3vu_virginia_edu/IgAgeeFylog8T62lp4NTkhNDAVaDEMGznPdSu2cUngmeDrQ?e=M5qECC)

## Pipeline

[Data Preparation Notebook](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/529f8f582e0adf04de670a3eb5b7d6b4b4464a00/data/project2_data_prep.ipynb)  
[Analysis Pipeline Notebook](pipeline/project2_pipeline.ipynb)  
[Analysis Pipeline Markdown](pipeline/project2_pipeline.md)

## License

[MIT License](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/85c8d3d572dc5cebf8b2b804db764a745765ad20/LICENSE)

---

## Problem Definition

### General Problem

Projecting athletic performance.

### Specific Problem

How well can pitcher-season Statcast pitch-quality metrics from the 2018–2021 MLB seasons predict season-level xERA for qualified pitchers?

### Rationale for Refinement

The original problem of projecting athletic performance is very broad, so this project narrows the scope to Major League Baseball pitchers and focuses specifically on expected ERA as the outcome of interest. This refinement makes the problem measurable, data-driven, and appropriate for a document-model pipeline because Statcast provides detailed pitch-level observations that can be stored as individual MongoDB documents.

Pitching is a strong refinement because pitcher performance depends heavily on repeatable process-based skills such as velocity, spin rate, pitch movement, pitch type, command, and contact quality allowed. By focusing on qualified pitchers from the 2018 through 2021 seasons, the project creates a manageable but meaningful dataset that supports both document storage and predictive modeling.

### Motivation

Traditional pitching statistics such as ERA can be useful, but they do not always isolate a pitcher’s actual skill because they can be influenced by defense, ballpark conditions, sequencing, and random variation in batted-ball outcomes. Statcast makes it possible to evaluate pitchers using the physical characteristics of each pitch, including velocity and spin rate, along with what happens when hitters make contact.

MLB’s expected metrics are especially useful because they aim to credit the pitcher for the quality of the event at the moment of contact rather than for downstream factors like defense or weather. This makes pitcher evaluation a strong setting for a data science project because the domain is rich, the data is detailed, and the problem naturally supports a document database and predictive pipeline.

### Headline of Press Release

[Statcast Data Reveals What Truly Drives Pitching Performance](docs/press_release.md)

---

## Domain Exposition

This project exists within the domain of sports analytics, specifically baseball pitching analysis. Modern baseball analysis relies heavily on Statcast, which tracks detailed pitch-level data such as velocity, spin rate, pitch type, release characteristics, and contact outcomes. These metrics allow analysts to evaluate pitchers based on the quality of their pitches rather than only on traditional results.

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

[Background readings folder](INSERT ONEDRIVE FOLDER LINK)

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

This project uses pitch-level MLB Statcast data from the 2018 through 2021 seasons. The raw data is collected in Python, filtered to include the variables needed for pitcher analysis, cleaned, and prepared for storage in MongoDB. Each observation in the final dataset represents one pitch, so each MongoDB document corresponds to a single pitch thrown during a regular-season MLB game.

After the raw Statcast data is loaded, the preparation process keeps pitcher-related fields such as pitcher ID, player name, game date, pitch type, release speed, release spin rate, count, inning, and contact-quality variables when available. Rows with missing values in critical fields are removed, column names are standardized, and each cleaned row is converted into a dictionary so it can be inserted into MongoDB as a document.

### Data Processing Pipeline

The data pipeline transforms raw Statcast pitch-level data into a document database and then into a modeling dataset.

The main steps are:

1. **Filtering**  
   The raw data is filtered to include regular-season pitch-level observations from the 2018 through 2021 MLB seasons.

2. **Feature Selection**  
   Pitcher-related variables are retained, including pitch type, velocity, spin rate, count, inning, launch speed, launch angle, and expected contact-quality metrics.

3. **Cleaning**  
   Rows with missing values in critical fields are removed. Column names are standardized so that MongoDB documents follow a consistent soft schema.

4. **Document Creation**  
   Each cleaned pitch-level row is converted into a dictionary, allowing it to be inserted into MongoDB as an individual document.

5. **MongoDB Storage**  
   The cleaned pitch-level records are stored in a MongoDB collection. This satisfies the document-model requirement and allows flexible querying.

6. **Modeling Table Construction**  
   The analysis pipeline queries MongoDB into a pandas dataframe and aggregates pitch-level observations into pitcher-season summaries for machine learning.

### Code Table

| File | Description | Link |
|---|---|---|
| `project2_data_prep.ipynb` | Loads raw Statcast data, keeps relevant pitcher variables, cleans missing values, standardizes column names, converts rows to MongoDB-ready documents, and inserts records into MongoDB | [notebook](https://github.com/devaswani2012/ds4320-pitcher-statcast-performance/blob/529f8f582e0adf04de670a3eb5b7d6b4b4464a00/data/project2_data_prep.ipynb) |
| `project2_pipeline.ipynb` | Queries MongoDB, converts documents into a dataframe, creates modeling features, trains a predictive model, evaluates results, and creates visualizations | [notebook](pipeline/project2_pipeline.ipynb) |
| `project2_pipeline.md` | Markdown export of the analysis pipeline notebook | [markdown](pipeline/project2_pipeline.md) |

### Rationale for Critical Decisions

Pitch-level data was chosen instead of player-level summaries because it preserves the full detail of Statcast observations and ensures the dataset contains well over 1,000 documents. This also makes the project a natural fit for MongoDB because each pitch can be stored as a flexible document with identifying information, pitch characteristics, game context, and outcome variables.

MongoDB was selected because the document model is well-suited for semi-structured sports tracking data. Different pitch events may contain slightly different fields depending on whether the ball was put in play, whether contact was made, or whether certain expected metrics are available. A document database can preserve this structure more naturally than a rigid table.

xERA was selected as the primary outcome because it better reflects pitcher skill than traditional ERA. ERA is affected by defense, sequencing, and randomness, while xERA attempts to measure expected run prevention based on strikeouts, walks, and quality of contact. This makes it a stronger target variable for a predictive model based on pitch-quality features.

### Bias Identification

Several sources of bias may be present in the data creation process. Selection bias can occur if the dataset includes only qualified pitchers, because injured players, relievers, and lower-volume pitchers may be excluded. Survivorship bias may also occur because pitchers who remain qualified across a season are more likely to be successful or healthy.

Measurement bias may exist because Statcast data relies on tracking technology that can produce small errors in variables such as velocity, spin rate, launch speed, and launch angle. Contextual bias may also arise because pitch outcomes depend on opposing hitters, defensive support, ballpark conditions, weather, and game situation, which are not fully captured by the selected features.

### Bias Mitigation

This project reduces bias by using multiple seasons of data from 2018 through 2021 rather than relying on a single year. The use of xERA also helps reduce the influence of defense and random variation because expected metrics focus more on the quality of the event than on the final result of the play.

The analysis also uses pitch-level sample sizes and aggregation to reduce the effect of random single-pitch outcomes. Exploratory analysis is used to identify missing values, outliers, and unusual observations. Filtering decisions are documented so that limitations are transparent and the results can be interpreted appropriately.

---

## Metadata

### Soft-Schema Guidelines

Each MongoDB document represents a single pitch thrown during an MLB regular-season game. Documents follow a consistent soft schema with fields for pitcher identification, game information, pitch characteristics, count context, and outcome-related variables.

Core identifying fields include pitcher ID, pitcher name, game date, and season. Pitch-level features include pitch type, release speed, release spin rate, balls, strikes, and inning. Contact-quality fields such as launch speed, launch angle, and estimated wOBA are included when available.

Numerical variables are stored as integers or floats, categorical variables are stored as strings, and dates are stored in a standardized date format. Missing or invalid values in critical fields are removed during data preparation to make the documents more consistent for querying and analysis.

### Data Summary

| Collection | Description | Unit of Observation |
|---|---|---|
| `statcast_pitch_level` | Pitch-level Statcast data for MLB pitchers from 2018 through 2021 | One document per pitch |

### Data Dictionary

| Feature | Type | Description | Example |
|---|---|---|---|
| `pitcher_id` | Integer | Unique identifier for the pitcher | `592789` |
| `pitcher_name` | String | Name of the pitcher | `Gerrit Cole` |
| `game_date` | String/Date | Date of the game | `2021-06-15` |
| `season` | Integer | MLB season | `2021` |
| `pitch_type` | String | Type of pitch thrown | `FF` |
| `velocity_mph` | Float | Pitch velocity in miles per hour | `96.4` |
| `spin_rate_rpm` | Float | Pitch spin rate in revolutions per minute | `2450` |
| `balls` | Integer | Number of balls in the count | `2` |
| `strikes` | Integer | Number of strikes in the count | `1` |
| `inning` | Integer | Inning in which the pitch was thrown | `5` |
| `estimated_woba` | Float | Expected weighted on-base value for the event | `0.315` |
| `launch_speed` | Float | Exit velocity of the batted ball | `102.3` |
| `launch_angle` | Float | Vertical launch angle of the batted ball | `18.5` |
| `events` | String | Outcome of the plate appearance | `strikeout` |
| `description` | String | Description of the pitch result | `swinging_strike` |

### Quantification of Uncertainty

| Feature | Source of Uncertainty | Interpretation |
|---|---|---|
| `velocity_mph` | Tracking measurement variation | Small differences may occur due to measurement precision |
| `spin_rate_rpm` | Tracking measurement variation and pitch classification | Spin rate varies naturally by pitch type and pitcher |
| `estimated_woba` | Model-based expected outcome | Reflects estimated contact quality rather than guaranteed outcome |
| `launch_speed` | Contact measurement variation | Only available when contact occurs |
| `launch_angle` | Contact measurement variation | Can vary greatly based on swing path and pitch location |

Pitch-level outcomes contain meaningful randomness. A well-executed pitch can still result in a hit, while a poorly located pitch may become an out because of hitter error or defensive positioning. Using many pitch-level documents helps reduce random variation, but uncertainty remains important when interpreting the model results.

---

## Data

The MongoDB database is stored in MongoDB Atlas. Login credentials are not included in this GitHub repository. They will be submitted separately in Canvas as required by the project instructions.

---

## Problem Solution Pipeline

The pipeline is implemented in a Jupyter notebook and exported as a markdown file. The notebook queries MongoDB, loads the documents into a dataframe, performs feature cleaning and aggregation, trains a machine learning model, evaluates the model, and creates a visualization comparing predicted and actual pitcher performance.

The final pipeline demonstrates that Statcast pitch-quality features can be used to create a document-based predictive workflow for pitcher xERA.
