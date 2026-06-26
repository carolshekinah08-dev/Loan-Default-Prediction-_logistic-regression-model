Predicting Borrower Delinquency Risk: A Logistic Regression Approach 
Writing Sample — Quantitative and Econometric Analysis 
Carol Shekinah 
 
1. Introduction and Motivation 
Lending institutions routinely need to assess the likelihood that a borrower will become delinquent on their obligations, both to price risk appropriately and to manage portfolio-level exposure. This note presents a logistic regression model that estimates the probability of borrower default using a cross-sectional dataset of 10,000 individual loan records sourced from Kaggle. The dataset includes borrower-level information on income, employment, homeownership, credit history, and loan characteristics. 
The objective of the exercise is twofold: first, to build a classification model that can flag higher-risk borrowers with reasonable accuracy; and second, to interpret the estimated coefficients in order to understand which borrower and credit-history characteristics are most strongly associated with default risk. The analysis is conducted in Python using pandas, scikit-learn, and seaborn, and follows a standard applied econometric workflow: data cleaning, exploratory analysis, model specification, estimation, and out-of-sample evaluation. 
2. Data and Methodology 
2.1 Outcome Variable 
The outcome of interest, target, is constructed as a binary indicator equal to 1 if a borrower recorded one or more delinquencies in the past two years (delinq_2y > 0), and 0 otherwise. In the sample, 14.24% of borrowers fall into the delinquent category, while 85.76% do not — a moderate class imbalance that is addressed below. 
2.2 Data Cleaning and Feature Construction 
Several identifier and post-outcome variables were dropped from the feature set: loan_Id, paid_total, paid_principal, paid_interest, issue_month, loan_purpose, and earliest_credit_line. These were excluded because they either carry no predictive content or would not be observable at the time a lending decision is made. Missing values in numeric columns were imputed using the column median, and categorical variables (state, homeownership, employment title, loan grade, loan status, etc.) were converted to dummy variables using one-hot encoding. The resulting design matrix contains 4,878 features across 8,000 training observations. 
2.3 Estimation Strategy 
The data were split into a training set (80%, n = 8,000) and a held-out test set (20%, n = 2,000) using stratified sampling on the target variable, preserving the 14.24% default rate in both partitions. All features were standardised (zero mean, unit variance) using a scaler fit on the training data only. 
A logistic regression model was estimated on the training data with class_weight='balanced', which re-weights the loss function inversely to class frequencies. This was a deliberate choice given the class imbalance: an unweighted model would be tempted to simply predict 'no default' for almost every observation and still achieve ~86% accuracy, while being practically useless for identifying risky borrowers.
3. Exploratory Data Analysis 
Before estimation, two simple bivariate checks were used to sense-check the data. 
Figure 1. Annual income by default status. Income distributions for the two groups overlap substantially, suggesting that income alone is unlikely to be a strong univariate predictor. 

Figure 2. Homeownership type by default risk. Mortgage holders make up the largest group in both the default and non-default categories, broadly reflecting the overall composition of the sample.
4. Model Results 
4.1 In-Sample Fit 
The model achieves a training accuracy of 100%. The class-balanced weighting ensures the model does not trivially predict the majority class, and the high training accuracy reflects strong in sample fit across both delinquent and non-delinquent borrowers. 
4.2 Coefficient Interpretation 
Because all predictors were standardised before estimation, the magnitude of each coefficient can be loosely interpreted as the change in log-odds of default associated with a one-standard deviation increase in that predictor, holding other variables fixed. The table below summarises the five largest positive and negative coefficients. 
Table 1. Strongest predictors of higher default risk 
Feature 
Coefficient
delinq_2y (number of delinquencies in the last 2 years) 
+6.035
num_collections_last_12m (number of debt collections in the last  12 months)
+0.269
state_KS (borrower is from Kansas) 
+0.233
state_VA (borrower is from Virginia) 
+0.157
num_active_debit_accounts (number of active debit accounts) 
+0.155



Table 2. Strongest predictors of lower default risk 
Feature 
Coefficient
months_since_last_delinq (months since last missed payment) 
-2.266
months_since_90d_late (months since last 90-day late payment) 
-0.422
account_never_delinq_percent (percentage of accounts with no  delinquency history)
-0.343
accounts_opened_24m (number of accounts opened in the last 24  months)
-0.233
total_debit_limit (total credit limit across all debit accounts) 
-0.198



The strongest positive predictor is delinq_2y (number of delinquencies in the last 2 years), which directly reflects a borrower's recent default history and is by far the most powerful signal in the model. Other risk-increasing features include recent collections activity (num_collections_last_12m: number of debt collections in the last 12 months), geographic indicators (state_KS: Kansas, state_VA: Virginia), and the number of active debit accounts (num_active_debit_accounts), which may proxy for broader financial complexity. 
On the protective side, months_since_last_delinq (months since last missed payment) carries the largest negative coefficient: a longer time since the most recent delinquency is strongly associated with lower current risk. Similarly, months_since_90d_late (months since last 90-day late payment) and account_never_delinq_percent (percentage of accounts with no delinquency history) are intuitive proxies for repayment discipline. 
5. Model Performance and Limitations 
5.1 Out-of-Sample Performance
On the held-out test set (n = 2,000), the model achieves a test accuracy of 100% and an ROC AUC of 1.00. Precision and recall are both 1.00 for both classes — the model correctly classifies every borrower in the test set. 


Figure 3. ROC curve (AUC = 1.00) and confusion matrix on the test set (n = 2,000). The model achieves perfect classification on the held-out data. 
5.2 Limitations and Possible Extensions 
• High-dimensional categorical encoding: one-hot encoding of high-cardinality fields such as employment title produces a design matrix with nearly 4,900 columns relative to 8,000 training observations, raising overfitting and multicollinearity concerns. A production version of this model would group rare categories, use target-encoding, or drop very sparse fields. 
• Linearity assumption: logistic regression assumes a linear relationship between the predictors and the log-odds of default. Tree-based or gradient-boosted models could be used as a benchmark to test whether non-linearities or interaction effects (e.g., between income and loan amount) materially improve predictive performance. 
• Threshold choice: the reported precision/recall figures correspond to the default 0.5 probability threshold. In a lending context, the optimal threshold should be driven by the relative cost of a false negative (missing a future default) versus a false positive (declining a creditworthy borrower), ideally set in consultation with the institution's risk and underwriting teams. 
• Generalisation: the model is trained and evaluated on a single Kaggle dataset. Performance on live loan applications from a different population or time period may differ, and the model should be validated on out-of-time or out-of-sample data before any operational deployment.
6. Conclusion 
This exercise illustrates a complete applied econometric workflow: defining an outcome of interest, cleaning and preparing a real-world loan dataset, addressing class imbalance through re weighting, and evaluating a logistic regression model on held-out data using accuracy, precision, recall, and ROC-AUC. The model achieves strong in-sample and out-of-sample performance, and recovers coefficient signs that are consistent with expectations about credit risk , borrowers with recent delinquency history are flagged as higher risk, while those with long clean repayment records are treated as lower risk. Future work could explore regularisation to address the high dimensional feature space and tree-based models as a non-linear benchmark.
