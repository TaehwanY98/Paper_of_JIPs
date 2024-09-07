
## Clustered Federated Learning Based on Number of Samples Mahalanobis Distance in Sequence Data for Medical Area
____

### Abstract
- In hospital, meta data is a private data from patients after diagnosis by doctor. So, in bad case, sniffer or hijacker can get important information from patients or hospital. For the reason, hospital data must be anonymized and protected by specific system for using the data. A hospital data is generally few, but several hospital data have more data in distributed condition. If hospital have system communicating among the hospitals, researcher can more efficiently train deep learning models. Therefore, this paper selected clustered federated learning (CFL) solution. In general CFL scenarios, proper cluster make model more grouped and personalized. However, previous CFL have model heterogeneous issues. This paper proposed Number of Sample Mahalanobis Distance (NSMD) solution. This solution can decrease model heterogeneity and increase clustering performance. In the experiment, this paper show that NSMD-CFL is more efficient in WESAD (wearable Stress Affect Detection) and IMDB (Internet Movie Database) dataset than cosine-based CFL.

### Introduction
- In recent years, researchers can aggregate meta information from medical devices by various sensors. The meta information includes user’s private data, so researchers should find way to use medical device’s data with privacy. For solving this issue, this paper proposes federated learning solution. Federated learning is AI (Artificial Intelligence) model training method in distributed condition. Federated learning is a method of training a real model by sharing only parameters with a specific server without sharing the user’s real data. Because of this federated learning features, user’s data can be protected with privacy and researchers are able to train specific AI model [1].
However, federated learning has limitations. This paper focuses on model heterogeneous issues and proposes clustering method based on Number of Samples Mahalanobis Distance (NSMD). Through this method, can solve model heterogeneous issues. This research is experimented by comparing Fed-Avg with our NSMD clustered FL method at WESAD (Wearable Stress and Affect Detection) and IMDB (Internet Movie Database) dataset. In Result of the experiment, our clustering method show that is better than cosine similarity-based clustering by high Silhouette score.

### Related Work
- Clustered Federated Learning
In recent, clustered federated learning in medical area proposed proper algorithm. Shiyi, Jiang et. al. proposed silhouette score based clustering algorithm [2][3]. And then Yoo Joo Hun et.al proposed personalized federated learning by using clustering method [4][5][6]. Although They also try to solve heterogeneous data condition, they have limitations about model parameters heterogeneity. Because they applied aggregation method by average method (Fed-Avg), FL applicant’s model parameters have each different importance in aggregation method. For the reason, there are heterogeneity among the model parameters. This type of heterogeneity affects the decrease performance of AI model.
For the detail, as well as general clustered federated learning has been proposed as a solution for how to improve performance in the presence condition in data heterogeneity, in this paper, proposed method can solve model heterogeneity issue in non-IID situations. Non-IID situations in which data and model heterogeneity occur can be divided into five major categories as shown in the following table.

| Non-IID case | Description and examples |
| :---- | :--- |
| Feature distribution skew |  Marginal distributions of data features differ. ex) Even if two individuals wear the same smartwatch model and exercise for the same time duration, the features of measured values are unique due to the personal characteristics difference, such as their gait     |
| Label distribution skew | Marginal distributions of data labels differ. ex) Frostbite is a disease that frequently occurs in cold areas because it is caused by exposure to severe cold resulting in tissue damage to body parts. Therefore, it is rare in places with relatively warm temperatures |
| Same label but different features| Conditional distributions of data features differ. ex) Medical devices are used to measure healthcare data such as neuro images and biomarkers of patients. However, hospitals do not use the identical medical device brands |
| Same feature but different labels | Conditional distributions of data labels differ. ex) Lung imaged by the recent pandemic COVID-19 virus are difficult to distinguish from the pneumonia because they have similar features in many lesions |
| Quantity skew | Amount of each patients/hospital data differs. ex) Suppose five times more patients have visited hospital A than hospital B. The quantity of data each hospital has will also significantly differ |

In this paper, a non-IID situation of a feature distribution skew was adopted to set a data heterogeneity environment. Through this condition, the performance of the clustered federated learning can be evaluated by solving heterogeneity issue.

### Experiments
- WESAD (Wearable Stress Affect Detection) Dataset

The WESAD dataset contains stress levels and metadata aggregated through experiments on specific wearable device’s users. Meta data collection was aggregated through wearable devices measured from the wrist and chest. Types of meta data measured from the chest include (ACC: 3-axis Accelerometer, ECG: Electrocardiogram, EMG: Electromyogram, EDA: Electrodynamic Activity, TEMP: Skin
Temperature, RESP: Respiration). All signals are sampled at 700 Hz [7]. This paper analyzes the WESAD data measured from the chest collected from specific users. The user’s all meta data has time-series data features. To select input data for detecting stress AI model training in condition which stress is output variable, input data and output data need to be calculated correlation index between them. Because correlation index shows the input variable’s effect to deep learning result and then inform importance of the variable in the training.

- Performance Table

| Model | Round/Epoch | Accuracy | F1-Score | Crossentropy Loss |
| :---: | :---: | :---: | :---: |  :---: |      
|Centralized Learning| x/30 | 76.89%| 45.26%| 0.5450|
|Fed-Avg| 30/2 | 76.48% |43.33%|0.6095|
|NSMD-FedAvg| 30/2 | 76.50% |43.34%|0.5330|

### Conclusion
- In this paper, NSMD method is proposed for solving model heterogeneous issues. This method is based on mahalanobis distance and normalizing model parameters with model heterogeneity. For the reason, client’s model parameter was decreased model heterogeneity. As a result, our method makes clustering method more efficient. In experiments, NSMD method show high silhouette score compared to cosine-based method in WESAD and IMDB. However, this method also has challenges likes using cluster weighted or other conditions about clustering method. And because this paper only selects LSTM model, the other attention-based model can increase performance.

____

Run Code

Step 1: Run Server

    python ClusteredFedAvgServer.py -v ClusteredFedAvg -w ./WESAD -i 1 -t False -r 30 -bs 1
In this code, batch_size must be set 1. Server Port is set 8084.

Step 2: Run Client
    
    python client.py -v ClusteredFedAvg -w ./WESAD -i 1 -t False -e 2 -bs 1

Step 3: Do Evaluation

    python evaluate.py -v ClusteredFedAvg -w ./WESAD
