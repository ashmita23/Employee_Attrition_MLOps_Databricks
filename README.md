### Problem Statement: 
Build a scalable machine learning pipeline in Databricks to predict employee attrition, ensuring the model's reliability over time through continuous monitoring for data drift. By integrating data ingestion, feature engineering, model training, and endpoint creation, the pipeline provides actionable insights into employee retention, while maintaining high prediction accuracy via robust model monitoring using Evidently AI.



The initial EDA of the data which was followed by binning of numerical features for better accuracy.
<img width="808" alt="Screenshot 2024-09-02 at 1 41 49 PM" src="https://github.com/user-attachments/assets/42e74030-965c-4a85-9cc0-dc24b355dbea">



### Modelling:

<img width="798" alt="Screenshot 2024-09-02 at 1 42 02 PM" src="https://github.com/user-attachments/assets/e49d745c-e6c9-43ad-91f3-acf6246bd395">

After conducting an initial run of AutoML, I was able to determine the optimal range of hyperparameters for the predictive modeling task. This automated approach helped in narrowing down the best-performing algorithms by exploring various models and their configurations. Among the models evaluated, XGBoost emerged as the top contender due to its high performance and robustness.

With XGBoost identified as the leading model, I proceeded to fine-tune it further using HyperOpt, a powerful optimization tool that systematically explores the hyperparameter space to enhance model performance. By applying HyperOpt to the XGBoost model, I was able to fine-tune the parameters, resulting in an even more accurate and efficient predictive model. 


<img width="453" alt="Screenshot 2024-09-02 at 1 42 23 PM" src="https://github.com/user-attachments/assets/a8a6f1fd-99cf-49f8-85e3-e5b59e8d689e">

<img width="735" alt="Screenshot 2024-09-02 at 1 42 49 PM" src="https://github.com/user-attachments/assets/b46c6ead-c57e-4b06-9087-a3310697df02">


### Databricks Workflow: 

<img width="804" alt="Screenshot 2024-09-02 at 1 42 59 PM" src="https://github.com/user-attachments/assets/ea07e5ec-f111-4778-80f5-e9c3d18c060c">


Developed an end-to-end machine learning pipeline for predicting employee attrition. The workflow begins with the ingestion of raw data directly into a SQL warehouse, followed by a query process to convert this data into a structured tabular format. The data is then pre-processed, and relevant features are engineered to be stored in a feature store, ensuring they are readily available for model training and inference. Using XGBoost, a powerful and scalable model, predictions are made on whether employees are likely to stay or leave the company. These predictions are then used to create a usable UI endpoint, making the results accessible and actionable for business stakeholders.

Additionally, the pipeline includes a robust model monitoring component utilizing Evidently AI. This step involves generating a monitoring report based on the test set inputs, which is crucial for detecting any potential data drift. A shift in the dataset is applied, and a subsequent prediction is made using the shifted data. The final step involves creating an endpoint to generate a report that identifies any drift, ensuring the model's predictions remain accurate and reliable over time. This comprehensive approach not only provides insights into employee attrition but also maintains the model's performance through continuous monitoring and adjustments.


