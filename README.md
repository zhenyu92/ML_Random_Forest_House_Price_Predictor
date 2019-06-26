# Chan Zhen Yue's Technical Assessment for AIAP
This project is a part of the evaluation of my application to AIAP.

### Project Status: [Completed]

### Project Objective
The study is on a market historical of real estate valuation collected from Sindian district, New Taipei City, Taiwan. Dataset is crawled from `https://aisgaiap.blob.core.windows.net/aiap4-assessment/real_estate.csv`.

### Environment Prerequisites
`Python3`, `pip3` & `curl` packages are required. To install, type the following in terminal:
```
$ sudo apt install python3 python3-pip curl
```

### Instruction
Run the executable bash script named `run.sh` at the base folder. 
```
bash run.sh
```
The script will:
- Install prerequisites library as stated in `requirements.txt`.
    ```
    numpy
    panda
    matplotlib
    sklearn
    seaborn
    ```
- Download the dataset as `real_estate.csv`.
- Run `AIAP.py`, a Python script which import `real_estate.csv` and perform Machine Learning to train a regression model.

### ML Methods Covered
There are two ML methods covered in this study:
* Linear Regression
* Random Forest Regresion
To select, key in the corresponding index:
```
Which model do you want to use? [1] Linear Regression, [2] Random Forest
```

### Exploratory Data Analysis
Detailed explanation can be found in eda.ipynb.
- 1 Importing relevant libraries
- 2 Loading raw data
- 3 Preprocessing
    * 3.1 Exploring the descriptive statistics of the variables
        + 3.2.1 Handling categorical variable
    * 3.2 Dealing with missing values
    * 3.3 Looking for correlation
    * 3.4 Exploring the PDFs
        + 3.4.1 Exploring variables (X1 - X6, Y)
    * 3.5 Dealing with outliers
    * 3.6 Log transformation
- 4 Prepare data for ML and create test set
    * 4.1 Declare inputs and targets
    * 4.2 Data scaling
    * 4.3 Train test split
- 5 Select and train a model
    * 5.1 Linear regression model
    * 5.2 Random forest regressor
- 6 Apply model on test set
    * 6.1 Test with linear regressor
    * 6.2 Test with RF regressor
        + 6.2.1 Grid Search (fine-tune)
        + 6.2.2 Apply model to test set

### Author
For any queries, Derrick can be contacted at zchan012@e.ntu.edu.sg.
