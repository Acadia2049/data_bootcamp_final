1. Introduction

This project aims to predict whether a user will recommend a game on Steam—one of the largest digital distribution platforms for PC games—by analyzing game attributes and user behavior.

To address this predictive task, we integrate three datasets:

    1.	Games Dataset: Game-specific features like release dates, prices, ratings, and review counts.

    2.	Users Dataset: User-level data, including the number of games owned and reviewed.

    3.	Recommendations Dataset: User-generated reviews, capturing engagement metrics (e.g., hours played, helpfulness scores) and the recommendation status (is_recommended), which serves as the target variable.

The key objective is to determine the likelihood of a game recommendation (True or False) using both:

    •	Game attributes: Pricing, discounts, ratings.

    •	User behavior: Hours played and review metrics.

To ensure a focused and meaningful analysis, the data underwent rigorous filtering, sampling, and cleaning. The final dataset comprises 219,933 reviews spanning the top 500 most-reviewed games released between 1998 and 2023.

2. Data Loading and Cleaning

2.1 Initial Data Overview

The raw datasets are large, containing millions of rows. An overview of each dataset revealed the following:

    •	Games Dataset: 50,872 games with attributes such as price, discount, platform compatibility, user ratings, and review counts.

    •	Users Dataset: Records for 14 million users, detailing the number of games owned and reviewed.

    •	Recommendations Dataset: 41 million user reviews capturing user engagement and recommendation status.

Given the dataset sizes, a systematic approach was adopted to reduce computational complexity while retaining relevant information.

2.2 Data Filtering

To streamline analysis and focus on meaningful insights, the following filters were applied:

    1.	Game Selection

    •	Selected the top 500 games with the highest number of user reviews.

    •	Focused on widely discussed and well-engaged titles.

    2.	Active User Selection

    •	Identified highly active users who reviewed at least 40 games.

    •	Filtered out casual reviewers for a robust analysis.

    3.	Recommendations Filtering

    •	Included only reviews for the top 500 games and from highly active users.

    •	Ensured alignment across datasets and relevance to the predictive task.

This filtering process reduced the dataset size to a more manageable subset of 1.46 million reviews.

2.3 Data Sampling

To further optimize computational performance, a 15% random sample of the filtered recommendations dataset was drawn. This step retained representativeness while improving analysis efficiency.

The resulting dataset contains 219,933 reviews, focusing on:

    •	The top 500 games.

    •	User behavior from highly active users.

2.4 Merging and Transformations

The following steps prepared the final dataset for analysis:

    •	Merged relevant attributes from the Games Dataset (e.g., price, discount, positive ratio, release date) into the filtered Recommendations data.

    •	Converted the release date into a standard datetime format to enable time-based analysis.

The cleaned dataset includes 12 features spanning game attributes, user behavior, and review metadata:

    •	Game Features: Price, discount percentage, positive rating ratio, and release year.

    •	User Behavior: Hours played, helpfulness votes, and recommendation status (is_recommended).

2.5 Final Cleaned Dataset Overview

After the data loading and cleaning process, the final dataset consists of:

    •	219,933 user reviews.

    •	500 games released between 1998 and 2023.

    •	Comprehensive details on game pricing, discounts, user engagement, and recommendation status.

Now this cleaned and structured dataset serves as the foundation for Exploratory Data Analysis (EDA) and the predictive modeling phase.

3. Descriptive Statistics

To better understand the cleaned dataset and uncover initial trends, summary statistics were generated for key numerical features: final game price, discount percentage, positive rating ratio, and hours played. These features provide insights into game pricing strategies, user satisfaction, and player engagement.

3.1 Summary of Key Features

The table below presents descriptive statistics for the selected features:

Feature	Mean	Standard Deviation	Min	25%	50% (Median)	75%	Max

Final Price ($)	20.59	16.43	0.00	7.99	19.99	30.00	70.00

Discount (%)	3.15	15.31	0.00	0.00	0.00	0.00	90.00

Positive Ratio (%)	89.10	8.64	26.00	86.00	91.00	95.00	98.00

Hours Played	76.82	140.96	0.00	8.80	24.80	72.80	999.90

3.2 Observations and Trends

    1.	Game Pricing

    •	The average final price of games is approximately $20.59, with a standard deviation of $16.43, reflecting significant variation in pricing.

    •	The median price is $19.99, indicating that most games cluster around mid-range pricing.

    •	Prices range from $0 (free-to-play games) to a maximum of $70, typical for premium AAA titles.

    2.	Discount Patterns

    •	A majority of games receive no discount, as reflected by a 0% discount at the 25th, 50th, and 75th percentiles.

    •	The maximum observed discount is 90%, suggesting occasional heavy price reductions during sales events.

    •	The low mean discount (3.15%) and high standard deviation (15.31) indicate that steep discounts are rare and selective.

    3.	Positive Rating Ratio

    •	Games in this dataset tend to be highly rated, with an average positive rating of 89.10%.

    •	The majority of games fall between 86% and 95% positive ratings, reflecting strong user satisfaction.

    •	The minimum observed positive ratio is 26%, highlighting a small subset of poorly received games.

    4.	User Engagement (Hours Played)

    •	The average playtime per game is 76.82 hours, but the high standard deviation of 140.96 indicates significant variability.

    •	The data is skewed: most users play games for fewer than 50 hours (25th percentile = 8.8 hours, median = 24.8 hours).

    •	A small subset of games shows extremely high user engagement, with playtimes nearing 1,000 hours.

3.3 Summary

The descriptive statistics reveal that:

    •	Game prices are predominantly mid-range, with free-to-play games forming a notable segment.

    •	Discounts are uncommon, but deep reductions occur selectively during sales.

    •	User satisfaction is high, as most games achieve positive ratings above 80%.

    •	Playtime varies widely, with most games engaging users for a moderate duration, while a few demonstrate exceptional long-term engagement.

These findings provide a foundational understanding of the dataset and will inform the subsequent Exploratory Data Analysis (EDA) and predictive modeling phases.

Then we start plotting:

![1734403629565](image/write_up/1734403629565.png)

The distribution of final game prices reveals a significant concentration at $0, emphasizing the popularity of free-to-play games among highly reviewed titles. Prices also tend to cluster around $10, $20, and $30, suggesting that mid-range pricing is a common strategy for game developers. Notably, fewer games are priced above $40, with the highest price reaching $70, indicating that premium-priced titles are less frequent and likely reserved for high-value or AAA games. This pattern highlights a balance between free-to-play models and mid-range pricing, with premium games forming a small, exclusive segment of the market.

![1734403672218](image/write_up/1734403672218.png)

The distribution of game discounts reveals that the majority of games receive no discount (0%), reflecting a strategy to maintain their base price. Only a small subset of games features discounts, with sporadic occurrences beyond the 20% mark. A slight increase is observed near 80%, likely tied to promotional events or heavy price cuts during sales. This pattern highlights that deep discounts are uncommon for top-reviewed games, as developers likely aim to preserve the perceived value of their titles.

![1734404100863](image/write_up/1734404100863.png)

The positive ratio distribution is skewed heavily toward higher values, with the majority of games achieving ratings above 80%. A notable concentration occurs between 90% and 95%, demonstrating that top-reviewed games are generally well-received. There is a sharp decline near 100%, indicating that while games are highly rated, achieving near-perfect scores is rare. This trend reinforces the notion that top-reviewed games consistently meet or exceed player expectations, reflecting their strong quality and widespread appeal on the Steam platform.

![1734404270428](image/write_up/1734404270428.png)

The distribution of hours played is highly skewed, with the majority of users playing for fewer than 50 hours. Playtime frequency decreases sharply as hours increase, but a small subset of users shows extreme engagement, with playtimes exceeding 200 hours and a few approaching 1,000 hours. This indicates that while most players interact with games for a moderate duration, a dedicated minority invests substantial time, reflecting a range of player engagement levels.

![1734404547762](image/write_up/1734404547762.png)

The Hours Played vs Positive Ratio plot shows that games with a higher positive ratio generally have a wide range of playtime, with many games achieving high ratings even at lower hours played. A significant concentration of points appears at positive ratios above 80%, regardless of playtime. However, games with extremely high playtime (e.g., 500+ hours) still maintain positive ratings, suggesting that sustained player engagement often correlates with user satisfaction. Lower positive ratios (below 50%) tend to cluster at lower playtime, implying that games with poor ratings fail to retain players for extended periods.

![1734404632233](image/write_up/1734404632233.png)

The Final Price vs Positive Ratio plot reveals that highly-rated games (positive ratio above 80%) are distributed across all price levels, including free-to-play games priced at $0. However, the concentration of high ratings is most notable at lower price points. As prices increase, the positive ratio becomes more varied, with occasional games receiving lower ratings. This suggests that while lower-priced or free-to-play games are consistently well-received, higher-priced games are subject to greater scrutiny and may have more polarized feedback.


## Conclusion

The analysis of top-reviewed Steam games reveals key trends in pricing, player engagement, and satisfaction. Free-to-play and mid-range priced games dominate, while premium-priced games ($40+) are less common. Discounts are rare, with most games maintaining their original price, reflecting a focus on preserving perceived value.

Player satisfaction is consistently high, with positive ratings skewed above 80% and peaking between 90–95%, highlighting strong overall game quality. Playtime is highly skewed, with most users playing under 50 hours, though a small subset invests 200+ hours, indicating high replay value for certain titles.

Games with higher positive ratings tend to sustain engagement across varying playtimes, while lower-rated games struggle to retain players for long durations. Similarly, lower-priced games receive more consistent positive feedback, whereas premium titles face greater scrutiny and polarized opinions.

In summary, affordability, quality, and player engagement are the key drivers of success in the competitive Steam gaming market.


**4. Modeling and Interpretations**

To predict whether a user will recommend a game, I employed multiple classification models and compared their performance. This process involved defining a robust  **baseline model** , developing a  **multiple regression model** , and evaluating their ability to capture patterns in the data.

**4.1 Approach and Workflow**

*The dataset was divided into an **80-20 train-test split** to ensure fair evaluation.

*Models were trained on **80% of the data** and tested on the remaining  **20%** , allowing performance to be evaluated on unseen data.

*The **Mean Squared Error (MSE)** was used as the evaluation metric. While MSE is typically used for regression, in this case, it quantifies the error when predicting the binary target **is_recommended**.

**4.2 Baseline Model**

To establish a performance benchmark, I created a simple **baseline model** that predicts the mean value of the target variable **is_recommended** for all instances. This approach serves as a reference point to determine the value added by more complex models.

***Baseline MSE** : **0.1326**

The baseline assumes no predictive power beyond the overall average recommendation rate. Any model that significantly reduces the MSE can be considered successful in capturing patterns and relationships within the data.

**4.3 Multiple Regression Model**

To improve upon the baseline, I developed a **multiple regression model** that incorporates game features and user behavior as predictors:

***Game Features** : **price_final** (final price), **discount** (percentage discount), and **positive_ratio** (positive user rating).

* **User Behavior** : **hours** (total hours played).

**Model Results**

The model was trained on the 80% training data and evaluated on the 20% test set. Key results include:

**Metric**	**Value**

**Training MSE**	 **0.1217**

**Testing MSE**	 **0.1247**

**Baseline MSE  0.1326**

The multiple regression model outperformed the baseline, achieving a **lower MSE** on both the training and testing datasets. This improvement indicates that the predictors successfully captured meaningful relationships with the recommendation outcome.

**Model Coefficients and Interpretations**

The model coefficients reveal the influence of each predictor on the target variable:

**Feature**	**Coefficient**	**Interpretation**

**positive_ratio 0.0118**	Strongest predictor: Higher user ratings increase recommendations.

**discount**	**0.0003**	Small positive effect: Discounts slightly increase recommendations.

**price_final**	**0.0002**	Minimal impact: Price has a weak but positive effect.

**hours**	         **0.0001**	Weakest predictor: Playtime has little influence on recommendations.

***1. Positive Ratio** : The strongest predictor, indicating that user satisfaction (measured through positive ratings) is the most significant driver of recommendations. A 1% increase in positive ratings leads to a notable increase in the likelihood of a game being recommended.

***2. Discount** and  **Price** : Discounts showed a small positive influence, suggesting that promotional pricing strategies contribute to recommendations but to a lesser extent. Interestingly, the effect of price (**price_final**) was minimal but positive, possibly reflecting the quality perception of higher-priced games.

***3. Hours Played** : The weakest predictor, implying that while playtime reflects engagement, it does not necessarily determine user satisfaction or recommendation likelihood. This could indicate that other factors, such as game quality or expectations, carry more weight.

**4.3 Summary of Findings**

The multiple regression model demonstrated clear improvements over the baseline by leveraging game features and user behavior data:

***Model Performance** : The training MSE of **0.1217** and testing MSE of **0.1247** indicate good generalization to unseen data.

***Predictor Influence** : Positive ratings (**positive_ratio**) emerged as the most impactful factor, reinforcing the critical role of user satisfaction in driving recommendations.

***Strategic Insights** : While discounts and pricing strategies play a role, their impact is limited. Developers and publishers should prioritize delivering high-quality games that resonate with users to maximize recommendations.

This analysis highlights the value of using multiple predictors to understand and forecast user recommendations, providing actionable insights for game developers, platform managers, and stakeholders seeking to optimize game design, pricing, and user engagement strategies.

**4.4 K-Nearest Neighbors Regression Model**

To further improve upon the previous models, I implemented a  **K-Nearest Neighbors (KNN) regression model** . KNN predicts outcomes based on the similarity of data points in the feature space, making it well-suited to capture **localized patterns** in the data. Given the clustering patterns observed in the visualizations of features such as **positive_ratio**, **hours**, and **discount**, KNN was expected to provide better performance by focusing on relationships between similar games.

**4.4.1 Model Implementation**

The KNN model uses the following predictors:

***Game Features** : **price_final** (final price), **discount** (percentage discount), **positive_ratio** (positive rating percentage).

***User Behavior** : **hours** (total hours played).

To ensure the model’s robustness and performance:

***1.** **Data Standardization** : All features were standardized to have a mean of 0 and a standard deviation of 1, ensuring that differences in feature scales do not bias the KNN predictions.

***2. Train-Test Split** : The dataset was split into an **80% training set** and a  **20% testing set** .

***3. Hyperparameter Tuning** : A grid search with **5-fold cross-validation** was performed to identify the optimal number of neighbors (**n_neighbors**).

**4.4.2 Hyperparameter Optimization**

Using GridSearchCV, the optimal value for **n_neighbors** was determined to be  **30** . This value was selected based on the lowest cross-validated Mean Squared Error (MSE), ensuring that the model achieved a balance between bias and variance.

**4.4.3 Model Performance**

The performance of the KNN model was evaluated on both the training and testing datasets, with the following results:

**Metric**	**Value**

**Training MSE**	 **0.1088**

**Testing MSE**	 **0.1184**

**Multiple Regression MSE**	**0.1247**

**Baseline MSE**	**0.1326**

***Training MSE** : **0.1088** — Slightly lower than the testing MSE, indicating minimal overfitting.

***Testing MSE** : **0.1184** — Outperformed both the baseline and the multiple regression model, highlighting the ability of KNN to generalize well on unseen data.

The KNN model demonstrated its strength in capturing **non-linear relationships** and localized influences within the data, which the linear regression model could not fully capture.

**4.4.4 Feature Importance**

To understand the contribution of each feature, **permutation importance** was calculated. The results are summarized below:

**Feature**	**Importance**	**Interpretation**

**positive_ratio 0.1693**	Most important feature: User satisfaction drives recommendations.

**hours**	          **0.1391**	Significant importance: Engagement reflects user preference.

**price_final**	**0.0487**	Lower impact: Price affects recommendations to a limited extent.

**discount**	**0.0035**	Minimal influence: Discounts have negligible impact.

**1. Positive Ratio** : Emerged as the most critical predictor, reinforcing its role as a primary driver of recommendations. Games with higher positive ratings are consistently more likely to be recommended.

**2. Hours Played** : Demonstrated substantial importance, indicating that games with higher playtime tend to receive recommendations. This suggests that user engagement often correlates with satisfaction.

**3. Price and Discounts** : While price had some influence, its impact was modest compared to **positive_ratio** and **hours**. Discounts contributed minimally, highlighting that promotional pricing strategies are less influential in driving recommendations.

**4.4.5 Summary of Findings**

The KNN model outperformed both the baseline and the multiple regression model by effectively capturing **complex, localized patterns** in the data. Key takeaways include:

***Performance Improvement** : The testing MSE of **0.1184** reflects strong generalization to unseen data, demonstrating KNN’s ability to capture relationships beyond linear trends.

***Feature Importance** : Positive ratings (**positive_ratio**) and user engagement (**hours**) emerged as the dominant factors, while price and discounts had limited influence.

* **Strategic Insights** : For game developers and publishers, prioritizing user satisfaction and fostering prolonged player engagement are crucial for driving positive recommendations.

Overall, the KNN model’s success highlights its suitability for recommendation tasks where localized relationships between data points play a significant role in influencing outcomes.


**5. Decision Tree Regression Model**

To further explore the non-linear relationships within the data, I employed a Decision Tree Regression Model. Decision trees are particularly effective for capturing complex patterns while providing an interpretable framework for understanding the decision-making process. This transparency allows us to identify which features most significantly influence whether a game is recommended.

**5.1 Rationale and Implementation**

The decision tree model evaluates splits in the data based on feature values, creating a series of “if-then” rules to predict outcomes. This approach is advantageous for its:

***Ability to capture non-linear relationships** : Unlike linear regression, decision trees adapt to the underlying structure of the data.

***Interpretability** : The tree structure visually demonstrates the importance and thresholds of features used in predictions.

To avoid overfitting, I experimented with various tree depths and regularization constraints to balance model complexity and generalization.

**5.2 Model Tuning and Performance**

***1. Optimal Tree Depth**

By iteratively testing tree depths from 1 to 20, I identified the optimal depth based on the Mean Squared Error (MSE) for the test set. The lowest MSE occurred at a depth of  **3** , suggesting a trade-off between simplicity and performance.

***2. Simplified Decision Tree**

To further improve generalization and interpretability, I trained a simplified decision tree with the following constraints:

**Max Depth** : 3

**Minimum Samples Split** : 100

**Minimum Samples Leaf** : 50

**Max Leaf Nodes** : 20

***3. Model Evaluation**

The performance of the simplified decision tree is summarized below:

**Metric**	**Value**

Training MSE    0.1173

Testing MSE      0.1199

KNN Testing MSE 0.1184

Baseline MSE     0.1326

The simplified decision tree **outperformed the baseline** but performed slightly worse than the KNN model. This difference likely stems from the tree’s depth constraint, which prevented it from fully capturing the complex, non-linear interactions in the data.

**5.3 Feature Importance**

To understand the contribution of each predictor, permutation importance was calculated for the simplified decision tree:

**Feature**	         **Importance**	**Interpretation**

positive_ratio	**0.1209**	*Dominant predictor: Higher user satisfaction drives recommendations.

hours	         **0.0898**	*Moderate impact: Engagement is a secondary driver of recommendations.

price_final	**0.0056**	*Minimal effect: Final price has limited influence on recommendations.

discount   	**0.0000**	*Negligible impact: Discounts do not meaningfully contribute to predictions.

**1.** **Positive Ratio** : Emerged as the most influential feature, reinforcing its critical role in determining recommendations. Games with high user satisfaction consistently received more positive reviews.

**2.** **Hours Played** : Displayed moderate importance, suggesting that while user engagement influences recommendations, its role is secondary to overall user satisfaction (positive_ratio).

**3. Price and Discounts** : Both features contributed minimally, with discounts showing no measurable importance. This result suggests that pricing strategies alone are insufficient to predict user recommendations, especially when user satisfaction is strong.

**5.4 Model Insights**

The simplified decision tree provides the following insights:

**Predictor Focus** : The model primarily relied on **positive_ratio** and  **hours** , ignoring weaker predictors like price and discounts. This simplified focus contributed to its interpretability but may have limited its predictive performance.

**Performance Trade-offs** : While the model achieved strong performance ( **testing MSE = 0.1199** ), it underperformed slightly compared to KNN. The constraints on tree depth likely prevented it from modeling finer details in the data.

**Interpretability** : The visual representation of the tree highlights the decision-making process, showing how key thresholds in **positive_ratio** and **hours** influence recommendations.

**5.5 Summary of Findings**

The decision tree model demonstrated competitive performance, providing an interpretable framework for understanding game recommendations:

**Positive Ratio** remains the most important factor, reaffirming that user satisfaction drives recommendations.

**Hours Played** plays a secondary role, highlighting the value of user engagement.

**Discounts and Price** contribute minimally, suggesting that users prioritize game quality over pricing strategies.

While the decision tree model did not outperform KNN, its transparency and ability to capture non-linear relationships make it a valuable tool for identifying key factors influencing game recommendations.


**6. Random Forest Regression Model**

As a final model, I extended the decision tree approach into a Random Forest Regression Model, an ensemble learning method that combines multiple decision trees to enhance predictive performance. Random forests reduce overfitting and improve generalization by averaging predictions from individual trees, making them a robust choice for this task.

**6.1 Rationale and Implementation**

Random forests build on the strengths of decision trees by:

**Combining predictions** from multiple trees to improve accuracy.

**Introducing randomness** to reduce variance and prevent overfitting.

Given the competitive performance of the single decision tree, the random forest was expected to capture additional complexities in the data while maintaining strong generalization.

**6.2 Model Tuning and Performance**

To optimize the model, a grid search with 5-fold cross-validation was used to fine-tune hyperparameters:

 **Number of Trees (n_estimators)** : 50, 100, and 150.

 **Max Depth (max_depth)** : 3, 4, 5, and 6.

The best parameters were found to be:

**Number of Trees** : 150.

**Max Depth** : 6.

The optimized random forest was then evaluated on both training and testing datasets:

**Metric**	**Value**

Training MSE                  0.1139

Testing MSE                    0.1169

KNN Testing MSE          0.1184

Decision Tree Testing MSE   0.1199

Baseline MSE                 0.1326

The random forest model achieved the **best performance** among all models, with a testing MSE of  **0.1169** , slightly better than the KNN model ( **MSE = 0.1184** ).

The training and testing MSEs were close, indicating strong generalization and minimal overfitting.

**6.3 Feature Importance**

The contribution of each predictor was assessed using permutation importance, which measures the decrease in model performance when a feature’s values are randomly shuffled. The results are summarized below:

**Feature**	         **Importance**	**Interpretation**

positive_ratio	**0.1309**	       Most significant predictor: User satisfaction drives recommendations.

hours	         **0.1188**	       Substantial influence: Engagement reflects user preferences.

price_final	**0.0138**	        Minor effect: Final price has a limited impact on recommendations.

discount   	**0.0002**	        Negligible contribution: Discounts do not meaningfully affect predictions.

**1.**	**Positive Ratio** : Confirmed as the dominant predictor, emphasizing the critical role of user satisfaction in driving recommendations.

**2.**	 **Hours Played** : Continued to show strong influence, indicating that engagement plays a vital role in determining recommendations.

**3.**	**Price and Discounts** : Both features contributed minimally, with discounts showing almost no impact on predictions. This result aligns with previous models, further emphasizing the importance of reviews and engagement over pricing strategies.

**6.4 Model Insights**

The random forest model highlights the advantages of ensemble methods:

**Improved Performance** : Achieved the **lowest testing MSE (0.1169)** among all models, demonstrating its ability to capture complex relationships in the data.

**Feature Stability** : Consistent results across models reinforced the importance of **positive_ratio** and **hours** as key predictors, while **price** and **discounts** remained secondary.

**Generalization** : The close alignment between training and testing MSEs confirmed the model’s robustness and suitability for unseen data.

**6.5 Summary of Findings**

The random forest model provided the most accurate and reliable predictions in this study. Key takeaways include:

 **Predictor Significance** : User satisfaction ( **positive_ratio** ) and engagement ( **hours** ) remain the primary drivers of recommendations.

 **Pricing Strategies** : Minimal contribution from **price_final** and **discount** suggests that pricing alone cannot drive positive recommendations without strong user satisfaction.

 **Practical Implications** : Developers and publishers should focus on improving game quality and fostering user engagement to maximize recommendations, while pricing strategies play a more supportive role.

Overall, the random forest model demonstrated the power of ensemble learning in improving predictive accuracy and provided actionable insights for understanding game recommendations on Steam.


#### Summary of Findings

In my analysis of game recommendations on Steam, all the models I developed demonstrated improved performance over the baseline predictor, indicating their utility in predicting user recommendations. The models ranked in terms of performance, based on their testing mean squared errors (MSE), are as follows: Random Forest Regression, K-Nearest Neighbors Regression (KNN), Decision Tree Regression, and Multiple Linear Regression.

Key Findings:

1. Success of the Random Forest Model: The Random Forest model emerged as the most effective, achieving the lowest testing mean squared error (MSE) of 0.1169, outperforming the K-Nearest Neighbors model (0.1184) and the Decision Tree model (0.1199). This performance highlights the model’s ability to capture complex, non-linear patterns within the data, while mitigating overfitting through its ensemble approach.
2. Impactful Features: Across all models, positive_ratio (percentage of positive reviews) and hours (user playtime) consistently emerged as the most important features influencing predictions: Random Forest feature importance: •positive_ratio: 0.1309 •hours: 0.1188 Conversely, price_final (final price) contributed marginally (0.0138) and discount had an almost negligible impact (0.0002).
3. Variable Influence Across Models: The consistent dominance of positive_ratio and hours across models demonstrates the strong influence of user sentiment and playtime on game recommendations. While price_final showed limited importance, discount had virtually no contribution across all models, suggesting that user behavior is driven more by engagement and reviews than by pricing or discounts.

## Next Steps & Discussion

Next Steps for Enhancing Predictive Models

To further improve the predictive capabilities of the models and gain deeper insights into game recommendations, I plan to incorporate the following additional features and analyses based on the provided data:

* User behavior and demographics will be added by incorporating user-specific features like the number of products owned and reviews made. This will help better understand user engagement levels and behaviors, which may influence whether a game is recommended. Users with more products or reviews may have a more critical perspective, impacting recommendation likelihood.
* Game ratings and release details will also be included. The rating column (e.g., Very Positive, Mixed) provides categorical insights into user sentiment and can serve as a strong predictor when encoded. The release date will allow for the extraction of temporal patterns, such as trends over time, where older games may have different recommendation rates compared to newer ones. Additionally, platform compatibility data, including Windows, Mac, Linux, and Steam Deck support, can reveal how accessibility impacts user recommendations.
* Review characteristics, such as the helpful and funny votes from user reviews, will also be analyzed. These metrics can uncover engagement levels with user reviews, which may indirectly reflect the quality of a game. Incorporating the review date allows for the analysis of temporal trends, capturing potential impacts of updates or recent events on user recommendations.
* Text feature extraction from game titles and descriptions will be explored using Natural Language Processing (NLP) techniques. Methods such as TF-IDF vectorization or word embeddings will help analyze keywords and recurring themes in game titles and descriptions. For instance, specific words like “Ultimate,” “Deluxe,” or “Simulator” could influence user recommendations and reveal trends in player preferences.
* Finally, financial and pricing data will be further analyzed by considering price patterns, such as price_original, price_final, and discount. Expanding this analysis over time will allow me to identify whether discounts or specific pricing strategies contribute to game recommendations. Additionally, incorporating financial data, such as average revenue per user if available, could further connect economic success with user recommendations.

By integrating user behavior, review characteristics, text-based analysis, and financial data, the models will become more comprehensive, offering deeper insights into the factors influencing game recommendations. These enhancements will refine the models’ predictive power, uncover hidden relationships within the data, and provide actionable insights for game developers and industry stakeholders.
