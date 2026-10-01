# Comment Category Prediction Challenge — MLP

## Overview
Welcome to the Comment Category Prediction Challenge! This dataset provides insight into how an online platform processes and categorizes user-generated comments. Each record represents a single comment and includes metadata such as interaction feedback, symbolic expressions, topic reference indicators, and internal system signals.

Your task is to explore this dataset and build predictive models that can accurately determine the final category assigned to each comment. Apply your analytical and modeling skills to uncover patterns across textual, numerical, and categorical features.

## Description
In this competition, you'll analyze a dataset containing information about short textual entries submitted to an online discussion system. The dataset includes the content of each entry, engagement signals from other users, and outputs from automated analysis components.

Your goal is to use this information to predict how each entry was ultimately categorized by the system.

## Dataset Description

### Data Files

| File | Description |
|---|---|
| `train.csv` | Training set, includes target variable `label` + all feature columns |
| `test.csv` | Test set, same feature columns but excludes target variable `label` |
| `sample_submission.csv` | Sample submission file in the correct format |

### Columns

| Column | Description |
|---|---|
| `comment` | The raw text content of the comment |
| `created_date` | The date and time when the comment was posted |
| `post_id` | A unique identifier linking the comment to a discussion thread or parent post |
| `emoticon_1` | Indicator for symbols from the first internal emoticon group |
| `emoticon_2` | Indicator for symbols from the second internal emoticon group |
| `emoticon_3` | Indicator for symbols from the third internal emoticon group |
| `upvote` | Number of positive reactions received by the comment |
| `downvote` | Number of negative reactions received by the comment |
| `if_1` | Internal feature one (hidden by the platform) |
| `if_2` | Internal feature two (hidden by the platform) |
| `race` | Whether the system detected references to a specific group identity |
| `religion` | Whether the system detected references to a belief-related topic |
| `gender` | Whether the system detected references to a gender-related topic |
| `disability` | Whether the system detected references to an ability-related topic |
| `label` | **Target Variable** — The final category assigned to the comment (4 distinct values) |

The official Kaggle data description defines four internal handling categories without assigning semantic names to label IDs. Previous semantic interpretations are unverified and have been removed. See [authenticated source findings](remote/README.md).

## Historical EDA notes (not independently reproduced)

| Label | `if_2` (median) | `if_1==0` rate |
|---|---|---|
| 0 | **4** | 79.3% |
| 1 | **10** | 17.0% |
| 2 | **10** | 76.6% |
| 3 | **10** | 79.6% |

- `if_2==4` is a strong indicator for Label 0
- `if_2==10` separates Labels 1/2/3 from Label 0
- Label 1: 20.3% black + 25.3% white race mentions vs ~4% for Label 0
- `if_1 > 0` is found in 83% of Label 1 samples

## Baseline Performance
- Leaderboard baseline: **0.65656** (macro F1)
- Previous v3 notebook: LightGBM + Logistic Regression ensemble
