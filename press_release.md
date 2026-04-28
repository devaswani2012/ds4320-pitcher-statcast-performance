# Statcast Data Reveals What Truly Drives Pitching Performance

## Hook

Why do some pitchers outperform others even when traditional statistics do not tell the full story? This project uses MLB Statcast data to look beneath the surface and evaluate pitchers based on the quality of the pitches they actually throw.

## Problem Statement

Traditional pitching statistics such as ERA are widely used, but they are not perfect measures of pitcher skill. ERA can be affected by defense, ballpark conditions, sequencing, luck, and other factors outside of a pitcher’s direct control. As a result, two pitchers with similar talent can end a season with very different traditional statistics.

This project focuses on a more specific question: how well can pitcher-season Statcast pitch-quality metrics from the 2018 through 2021 MLB seasons predict season-level xERA for qualified pitchers?

## Solution Description

The solution uses Statcast-derived pitcher-season data to build a MongoDB document database where each modeling document represents one pitcher in one MLB season. These documents include pitcher identifiers, season, xERA, average estimated wOBA allowed, plate appearances, fastball velocity, and fastball spin rate.

The analysis pipeline queries the MongoDB collection into Python, converts the documents into a dataframe, and trains a machine learning model to predict season-level xERA. Because xERA is designed to reflect expected performance rather than only actual outcomes, it provides a better target for evaluating pitcher skill.

## Chart

The main project visualization compares actual xERA against predicted xERA. This chart shows how closely the model’s predictions align with observed pitcher-season performance and helps evaluate whether Statcast pitch-quality metrics provide meaningful predictive value.

<img width="1470" height="834" alt="press_release_figure" src="https://github.com/user-attachments/assets/629e3310-67eb-4499-acde-0b548c1cfa73" />


*Figure 1: Three-panel figure. Top-left: distribution of xERA across the pitcher-season observations in the final modeling dataset, with the dataset mean marked. Top-right: actual versus predicted xERA from the Random Forest model, with the 45-degree line representing perfect predictions. Bottom: average pitcher xERA by season from 2018 to 2021.*

The figure connects the project narrative to measurable evidence: pitcher xERA varies meaningfully across qualified pitcher-seasons, Statcast-derived pitch-quality variables contain predictive signal, and league-level pitcher performance remains relatively stable across seasons even though individual pitchers differ substantially.

**Top-left — xERA distribution:** This panel shows the distribution of xERA in the final filtered dataset. Most pitcher-seasons fall within a moderate xERA range, with fewer observations at the extremes. The dashed line marks the dataset mean. This demonstrates that pitcher performance is concentrated around a typical league level, but still has enough variation to motivate predictive modeling.

**Top-right — Actual vs. predicted xERA:** This panel evaluates model performance by comparing actual xERA to predicted xERA on the held-out test set. The dashed 45-degree line represents perfect predictions. Observations closer to the line are better predicted by the model. The figure shows that Statcast-based metrics provide strong predictive value, while also making clear that some unexplained variation remains.

**Bottom — Average xERA by season:** This panel shows average xERA across seasons from 2018 through 2021. League-level averages remain relatively stable over time, suggesting that the main analytical challenge is not explaining dramatic league-wide changes, but rather understanding differences in performance across pitchers within the same season.

**Takeaway:** Pitcher xERA is not random. Statcast features such as expected wOBA, fastball velocity, fastball spin, and workload contribute meaningful predictive information. However, model performance should be interpreted carefully because expected wOBA is conceptually related to xERA and may partially reflect similar underlying information.
