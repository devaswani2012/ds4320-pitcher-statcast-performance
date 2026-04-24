# Statcast Data Reveals What Truly Drives Pitching Performance

## Hook

Why do some pitchers outperform others even when traditional statistics do not tell the full story? This project uses MLB Statcast data to look beneath the surface and evaluate pitchers based on the quality of the pitches they actually throw.

## Problem Statement

Traditional pitching statistics such as ERA are widely used, but they are not perfect measures of pitcher skill. ERA can be affected by defense, ballpark conditions, sequencing, luck, and other factors outside of a pitcher’s direct control. As a result, two pitchers with similar talent can end a season with very different traditional statistics.

This project focuses on a more specific question: how well can pitcher-season Statcast pitch-quality metrics from the 2018 through 2021 MLB seasons predict season-level xERA for qualified pitchers?

## Solution Description

The solution uses pitch-level Statcast data to build a MongoDB document database where each document represents a single pitch. These documents include pitch characteristics such as velocity, spin rate, pitch type, count, and contact-quality variables.

The analysis pipeline queries the MongoDB collection into Python, converts the documents into a dataframe, aggregates pitch-level records into pitcher-season features, and trains a machine learning model to predict xERA. Because xERA is designed to reflect expected performance rather than only actual outcomes, it provides a better target for evaluating pitcher skill.

## Chart

The main project visualization compares actual xERA against predicted xERA. This chart shows how closely the model’s predictions align with observed pitcher-season performance and helps evaluate whether Statcast pitch-quality metrics provide meaningful predictive value.

## Impact

This project shows how a document database can support sports analytics by preserving detailed pitch-level data while still allowing the information to be transformed into a predictive modeling dataset. The results can help analysts, coaches, and fans better understand which pitch characteristics are most closely connected to pitcher performance.
