Steam Game Recommendation Predictor

Table of Contents
	•	Introduction
	•	Version History
	•	Version 1.0
	•	Version 2.0
	•	Dataset Description
	•	Data Processing
	•	Data Loading and Cleaning
	•	Descriptive Statistics
	•	Exploratory Data Analysis
	•	Modeling and Methods
	•	Logistic Regression
	•	Random Forest Classifier
	•	Gradient Boosting Classifier
	•	Results and Interpretation
	•	Model Performance
	•	Feature Importance
	•	Cross-Validation
	•	Conclusion and Next Steps
	•	Repository Structure
	•	Usage
	•	Contributing
	•	License
	•	Contact

Introduction

Welcome to the Steam Game Recommendation Predictor project! This project aims to predict whether a user will recommend a game on Steam—one of the largest digital distribution platforms for PC games—by analyzing various game attributes and user behavior data.

Version History

Version 1.0

Release Date: [Dec.15th, 2024]

Key Features:
	•	Data Integration: Combined three datasets—Games, Users, and Recommendations—to create a comprehensive dataset.
	•	Data Processing: Implemented data loading, cleaning, filtering, sampling, and merging to prepare the final dataset.
	•	Descriptive Statistics: Generated summary statistics for key numerical features to understand data distributions.
	•	Basic Modeling: Developed initial classification models using Logistic Regression, Random Forest, and Gradient Boosting classifiers.
	•	Initial Insights: Identified positive review ratios and hours played as significant predictors of game recommendations.

Version 2.0

Release Date: [Dec.20th, 2024]

Enhancements and Additions:
	•	Advanced Data Processing: Incorporated additional temporal features and one-hot encoding for categorical variables to enhance model performance.
	•	Comprehensive Exploratory Data Analysis (EDA): Added detailed visualizations and insights, including distribution plots, correlation matrices, and density plots.
	•	Hyperparameter Tuning: Optimized Random Forest and Gradient Boosting classifiers using GridSearchCV to improve predictive accuracy.
	•	Enhanced Model Evaluation: Introduced cross-validation techniques to ensure model robustness and reliability.
	•	Feature Importance Analysis: Provided in-depth analysis of feature contributions across different models to guide strategic decisions.
	•	Improved Documentation: Updated README to include detailed methodology, findings, and strategic implications for stakeholders.

Dataset Description

The project integrates three primary datasets to facilitate comprehensive analysis:
	1.	Games Dataset:
	•	Size: 50,872 games
	•	Features: app_id, price_final, discount, platforms, positive_ratio, date_release, rating, and other relevant attributes.
	2.	Users Dataset:
	•	Size: 14 million users
	•	Features: user_id, games_owned, reviews, and other user-specific data.
	3.	Recommendations Dataset:
	•	Size: 41 million user reviews
	•	Features: review_id, app_id, user_id, helpful, funny, date, is_recommended, hours, and other engagement metrics.

After rigorous filtering, sampling, and cleaning, the final dataset comprises 35,445 reviews from the top 500 most-reviewed games released between 1998 and 2023.

Data Processing

Data Loading and Cleaning

Initial Data Overview
	•	Games Dataset: 50,872 games with various attributes.
	•	Users Dataset: 14 million user records.
	•	Recommendations Dataset: 41 million user reviews.

Given the large size, the data was filtered to focus on the top 500 games and highly active users, reducing the dataset to 236,303 reviews. A further 15% random sampling resulted in 35,445 reviews for analysis.

Data Filtering
	1.	Game Selection:
	•	Selected the top 500 games based on the highest number of user reviews.
	•	Focused on widely discussed and well-engaged titles.
	2.	Active User Selection:
	•	Identified highly active users who reviewed at least 24 reviews (99th percentile).
	•	Filtered out casual reviewers for a robust analysis.
	3.	Recommendations Filtering:
	•	Included only reviews for the top 500 games and from highly active users.
	•	Applied a helpfulness threshold of 4 helpful votes (90th percentile).

Data Sampling

To optimize computational performance, a 15% random sample of the filtered recommendations dataset was drawn, retaining 35,445 reviews for analysis.

Merging and Transformations
	•	Merged game attributes (price_final, discount, positive_ratio, date_release, rating) into the filtered recommendations data based on app_id.
	•	Converted date_release into a standard datetime format.
	•	Extracted additional temporal features: release_year, release_month, time_since_release.
	•	Encoded categorical variables (rating) using one-hot encoding.

The final dataset includes 13 features spanning game attributes, user behavior, and review metadata.

Descriptive Statistics

Summary statistics were generated for key numerical features:

Feature	Mean	Std Dev	Min	25%	Median	75%	Max
Final Price ($)	22.58	17.82	0.00	9.99	20.00	30.00	70.00
Discount (%)	1.92	12.49	0.00	0.00	0.00	0.00	90.00
Positive Ratio (%)	88.87	9.61	26.00	86.00	91.00	96.00	98.00
Hours Played	129.92	195.09	0.00	14.30	46.30	149.50	1000.00

Key Observations
	•	Pricing: Predominantly mid-range with a significant number of free-to-play games.
	•	Discounts: Rare and selective, often during sales events.
	•	User Satisfaction: High positive ratings above 80%.
	•	Engagement: Highly skewed playtime with most users playing fewer than 200 hours.

Exploratory Data Analysis

Comprehensive EDA was conducted to uncover initial trends and relationships within the data. Key visualizations and insights include:

Distribution of User Reviews

	•	50th Percentile: 1 review
	•	75th Percentile: 3 reviews
	•	90th Percentile: 6 reviews
	•	95th Percentile: 9 reviews
	•	99th Percentile: 24 reviews

Helpfulness Distribution

	•	50th Percentile: 0 helpful votes
	•	75th Percentile: 0 helpful votes
	•	90th Percentile: 4 helpful votes
	•	95th Percentile: 8 helpful votes
	•	99th Percentile: 49 helpful votes

Key Plots and Insights
	1.	Distribution of Recommendation Status:
	•	True: ~74.5%
	•	False: ~25.5%
	•	Insight: Majority of reviews are positive, indicating high user satisfaction.
	2.	Final Game Prices:
	•	Most games priced around $20.
	•	Significant number of free-to-play games.
	•	Insight: Mid-range pricing is prevalent, with premium titles being less common.
	3.	Discounts:
	•	Majority of games have 0% discount.
	•	Few games have high discounts (up to 90%).
	•	Insight: Discounts are rare and typically tied to promotional events.
	4.	Positive Review Ratios:
	•	Heavily skewed towards high values (80%+).
	•	Few games have low positive ratios.
	•	Insight: High user satisfaction is common among top-reviewed games.
	5.	Hours Played:
	•	Highly skewed distribution with most users playing <200 hours.
	•	Some users invest up to 1000 hours.
	•	Insight: Sustained engagement correlates with higher recommendation rates.
	6.	Rating and Recommendation:
	•	Higher ratings (“Very Positive”, “Overwhelmingly Positive”) correlate strongly with recommendations.
	•	Even some lower-rated games receive notable recommendations.
	•	Insight: User sentiment plays a critical role, though other factors may influence recommendations.
	7.	Correlation Matrix:
	•	Strong positive correlation between helpful and funny metrics (0.60).
	•	Moderate correlation between is_recommended and positive_ratio (0.32).
	•	Insight: Minimal multicollinearity, ensuring robust model performance.

Modeling and Methods

To predict whether a user will recommend a game, multiple classification models were employed and compared based on their performance metrics.

Model Selection
	1.	Logistic Regression: Baseline linear model to establish initial predictive capabilities.
	2.	Random Forest Classifier: Ensemble method leveraging multiple decision trees for improved accuracy and robustness.
	3.	Gradient Boosting Classifier: Advanced ensemble technique focusing on sequential model building to minimize errors.

Data Preparation
	•	Feature Selection:
	•	Features: helpful, funny, hours, price_final, positive_ratio, and one-hot encoded rating variables.
	•	Target: is_recommended.
	•	Data Splitting:
	•	Train-Test Split: 80-20 stratified split to maintain class balance.
	•	Scaling:
	•	Applied StandardScaler to normalize feature ranges, ensuring optimal model performance.

Logistic Regression
	•	Implementation:
	•	Utilized LogisticRegression from scikit-learn with max_iter=1000.
	•	Training:
	•	Trained on 80% of the data.
	•	Evaluation:
	•	Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

Random Forest Classifier
	•	Implementation:
	•	Utilized RandomForestClassifier from scikit-learn with default parameters initially.
	•	Hyperparameter Tuning:
	•	Conducted GridSearchCV with a parameter grid focusing on n_estimators, max_depth, min_samples_split, and min_samples_leaf.
	•	Best Parameters: n_estimators=300, max_depth=10, min_samples_split=2, min_samples_leaf=4.
	•	Training:
	•	Trained with optimized hyperparameters.

Gradient Boosting Classifier
	•	Implementation:
	•	Utilized GradientBoostingClassifier from scikit-learn with default parameters initially.
	•	Hyperparameter Tuning:
	•	Conducted GridSearchCV with a parameter grid focusing on n_estimators, learning_rate, max_depth, and subsample.
	•	Best Parameters: n_estimators=300, learning_rate=0.01, max_depth=7, subsample=0.8.
	•	Training:
	•	Trained with optimized hyperparameters.

Results and Interpretation

Model Performance

Model	Training MSE	Testing MSE	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.1217	0.1247	0.7651	0.7714	0.9731	0.8606	0.7361
Random Forest Classifier	0.1173	0.1199	0.7468	0.8067	0.8682	0.8363	0.7267
Gradient Boosting Classifier	0.1173	0.1199	0.7468	0.8067	0.8682	0.8363	0.7267
Tuned Random Forest	0.1139	0.1169	0.7811	0.7996	0.9424	0.8651	0.7767
Tuned Gradient Boosting	0.1139	0.1169	0.7797	0.7998	0.9394	0.8640	0.7764

	•	Random Forest and Gradient Boosting initially outperformed the Logistic Regression model.
	•	Hyperparameter Tuning significantly improved the performance of both ensemble models, achieving the highest Accuracy (~78%) and ROC-AUC (~0.776).
	•	Logistic Regression showed strong recall for the True class but struggled with the False class, indicating a bias towards predicting positive recommendations.

Classification Reports

Logistic Regression
	•	False Class:
	•	Precision: 0.67
	•	Recall: 0.16
	•	F1-Score: 0.25
	•	True Class:
	•	Precision: 0.77
	•	Recall: 0.97
	•	F1-Score: 0.86

Tuned Random Forest Classifier
	•	False Class:
	•	Precision: 0.65
	•	Recall: 0.31
	•	F1-Score: 0.42
	•	True Class:
	•	Precision: 0.80
	•	Recall: 0.94
	•	F1-Score: 0.87

Tuned Gradient Boosting Classifier
	•	False Class:
	•	Precision: 0.64
	•	Recall: 0.31
	•	F1-Score: 0.42
	•	True Class:
	•	Precision: 0.80
	•	Recall: 0.94
	•	F1-Score: 0.86

Feature Importance

Tuned Random Forest

Feature	Importance
positive_ratio	0.1309
hours	0.1188
price_final	0.0138
rating_Overwhelmingly Positive	0.0002

Tuned Gradient Boosting

Feature	Importance
positive_ratio	0.1309
hours	0.1188
price_final	0.0138
rating_Overwhelmingly Positive	0.0002

	•	Positive Ratio and Hours Played consistently emerge as the most influential features across both models.
	•	Price Final and Rating have minimal impact, indicating that user satisfaction and engagement are more critical in driving recommendations.

Cross-Validation

Model	ROC-AUC Scores	Mean ROC-AUC	Std Dev
Logistic Regression	[0.7446, 0.7157, 0.7183, 0.7210, 0.7334]	0.7266	0.0109
Random Forest	[0.7538, 0.7362, 0.7418, 0.7306, 0.7511]	0.7427	0.0088
Gradient Boosting	[0.7866, 0.7713, 0.7715, 0.7650, 0.7733]	0.7736	0.0071

	•	Gradient Boosting exhibits the highest cross-validated ROC-AUC scores, indicating superior ability to distinguish between classes.
	•	Random Forest follows closely, outperforming Logistic Regression consistently.
	•	Logistic Regression lags behind ensemble methods but provides a useful baseline.

Conclusion and Next Steps

Summary of Models
	•	Tuned Random Forest Classifier emerged as the most effective model, achieving the highest ROC-AUC and balanced performance metrics.
	•	Tuned Gradient Boosting Classifier closely followed, demonstrating robust predictive capabilities.
	•	Logistic Regression serves as a solid baseline, highlighting the significance of feature selection and model complexity in predictive performance.

Key Insights
	1.	User Satisfaction: High positive review ratios (positive_ratio) are the primary drivers of game recommendations.
	2.	User Engagement: Playtime (hours) significantly influences recommendations, indicating that sustained engagement correlates with user satisfaction.
	3.	Pricing Strategies: Minimal impact of price and discounts suggests that quality and engagement outweigh pricing in driving recommendations.

Strategic Implications
	•	Enhancing Positive Feedback: Encourage and facilitate positive user reviews to boost recommendation rates.
	•	Sustaining Engagement: Design engaging game mechanics to promote longer playtimes, fostering user loyalty and recommendations.
	•	Optimal Pricing Strategies: Maintain moderate pricing tiers, particularly around the $20 mark, which appears favorable for user endorsements.

Addressing Model Limitations
	•	Class Imbalance: The models exhibit challenges in accurately predicting non-recommended games. Strategies such as resampling techniques (e.g., SMOTE), adjusting class weights, or exploring alternative algorithms could mitigate this issue.
	•	Feature Engineering: Incorporate additional features or interactions that may better distinguish non-recommended games, potentially improving model recall and precision for the False class.

Next Steps for Enhancing Predictive Models
	1.	Incorporate Additional Features:
	•	User-Specific Data: Number of products owned, number of reviews made to better understand user engagement levels.
	•	Game Ratings and Release Details: Categorical insights into user sentiment, temporal patterns based on release dates.
	•	Platform Compatibility: Impact of accessibility (Windows, Mac, Linux, Steam Deck) on recommendations.
	•	Review Characteristics: Analyze helpful and funny votes, review dates for temporal trends.
	•	Text Feature Extraction: Utilize NLP techniques (e.g., TF-IDF, word embeddings) on game titles and descriptions to uncover influential keywords.
	•	Financial Data: Analyze price patterns (price_original, price_final, discount) and average revenue per user if available.
	2.	Advanced Modeling Techniques:
	•	Ensemble Methods: Explore stacking or blending different models to enhance predictive performance.
	•	Deep Learning Models: Implement neural networks for more complex feature interactions and representations.
	•	Cross-Validation Enhancements: Utilize more robust cross-validation techniques to ensure model robustness.
	3.	Model Deployment and Monitoring:
	•	Develop a user-friendly application or dashboard for real-time predictions.
	•	Integrate with Steam APIs for continuous data updates and model retraining to maintain accuracy over time.
	4.	Addressing Class Imbalance:
	•	Implement techniques such as SMOTE, undersampling, or cost-sensitive learning to improve model performance on the False class.
	5.	Continuous Feature Engineering:
	•	Continuously explore and incorporate new features that may influence recommendations, adapting to evolving user behaviors and market trends.

By leveraging these insights and refining the modeling approach, developers and marketers can strategically enhance game offerings, optimize pricing, and improve user engagement to maximize recommendation rates on Steam.

Repository Structure

steam-game-recommendation-predictor/
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── selected_games.csv
│   │   ├── active_users.csv
│   │   ├── filtered_recommendations.csv
│   │   ├── sampled_recommendations.csv
│   │   └── final_dataset.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Modeling.ipynb
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore

	•	data/: Contains raw and processed datasets.
	•	notebooks/: Jupyter notebooks for Exploratory Data Analysis (EDA) and modeling.
	•	src/: Source code for data processing, model training, and utility functions.
	•	images/: Directory for storing plots and visualizations used in the write-up.
	•	README.md: Project overview and documentation.
	•	requirements.txt: List of dependencies.
	•	LICENSE: Licensing information.
	•	.gitignore: Files and directories to ignore in version control.

Usage

Prerequisites
	•	Python 3.7+
	•	Jupyter Notebook
	•	Required Python packages listed in requirements.txt

