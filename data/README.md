# Data Folder

This folder is reserved for data-related documentation and small sample files.

The full dataset is stored in MongoDB Atlas rather than directly in GitHub. The project uses pitch-level MLB Statcast data from the 2018 through 2021 seasons. Each MongoDB document represents one pitch thrown during a regular-season MLB game.

Database credentials are not stored in this repository. Access information will be submitted separately through Canvas.

## Processed Data

The final modeling dataset is constructed by:

- Combining all Statcast pitch-level datasets (2018–2021)
- Filtering to regular-season MLB pitches
- Retaining pitcher-relevant variables:
  - Pitch type
  - Velocity (mph)
  - Spin rate (rpm)
  - Count (balls, strikes)
  - Inning
  - Contact-quality metrics (launch speed, launch angle, estimated wOBA)
- Removing rows with missing values in critical features
- Standardizing column names for consistency across seasons
- Converting each pitch-level observation into a MongoDB document
- Storing the cleaned dataset in a MongoDB collection

For modeling:

- Querying MongoDB and converting documents into a pandas dataframe
- Aggregating pitch-level data to the pitcher-season level
- Creating pitcher-season features:
  - Average velocity
  - Average spin rate
  - Average launch speed allowed
  - Average launch angle allowed
  - Average estimated wOBA allowed
- Constructing outcome variable:
  - Season-level xERA
- Filtering to qualified pitchers:
  - Minimum pitch count threshold to ensure stability
- Final modeling dataset consists of pitcher-season observations suitable for regression modeling

---

## Data Source

Statcast data is retrieved from Major League Baseball’s Statcast system via publicly available sources.

Sources include:

- Baseball Savant (https://baseballsavant.mlb.com/)
- Public Statcast datasets and Python interfaces (e.g., pybaseball)

Data accessed and processed: April 2026
