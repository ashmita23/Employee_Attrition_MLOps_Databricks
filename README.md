### Problem Statement: 
Build a scalable machine learning pipeline in Databricks to predict employee attrition, ensuring the model's reliability over time through continuous monitoring for data drift. By integrating data ingestion, feature engineering, model training, and endpoint creation, the pipeline provides actionable insights into employee retention, while maintaining high prediction accuracy via robust model monitoring using Evidently AI.



The initial EDA of the data which was followed by binning of numerical features for better accuracy.
<img width="808" alt="Screenshot 2024-09-02 at 1 41 49 PM" src="https://github.com/user-attachments/assets/42e74030-965c-4a85-9cc0-dc24b355dbea">



### Modelling:
After an initial run of AutoML to determine the range of hyperparameters, I ran HyperOpt on the best perfoming model and chose the best performing model for the predictive modelling.

<img width="798" alt="Screenshot 2024-09-02 at 1 42 02 PM" src="https://github.com/user-attachments/assets/e49d745c-e6c9-43ad-91f3-acf6246bd395">


<img width="453" alt="Screenshot 2024-09-02 at 1 42 23 PM" src="https://github.com/user-attachments/assets/a8a6f1fd-99cf-49f8-85e3-e5b59e8d689e">

### Process: 

Developed an end-to-end machine learning pipeline for predicting employee attrition. The workflow begins with the ingestion of raw data directly into a SQL warehouse, followed by a query process to convert this data into a structured tabular format. The data is then pre-processed, and relevant features are engineered to be stored in a feature store, ensuring they are readily available for model training and inference. Using XGBoost, a powerful and scalable model, predictions are made on whether employees are likely to stay or leave the company. These predictions are then used to create a usable UI endpoint, making the results accessible and actionable for business stakeholders.

Additionally, the pipeline includes a robust model monitoring component utilizing Evidently AI. This step involves generating a monitoring report based on the test set inputs, which is crucial for detecting any potential data drift. A shift in the dataset is applied, and a subsequent prediction is made using the shifted data. The final step involves creating an endpoint to generate a report that identifies any drift, ensuring the model's predictions remain accurate and reliable over time. This comprehensive approach not only provides insights into employee attrition but also maintains the model's performance through continuous monitoring and adjustments.

<img width="804" alt="Screenshot 2024-09-02 at 1 42 59 PM" src="https://github.com/user-attachments/assets/ea07e5ec-f111-4778-80f5-e9c3d18c060c">

