# Goodreads Book Recommendation System - PySpark & ALS

This project aims to develop a book recommendation system using PySpark and the Alternating Least Squares (ALS) algorithm on the [Goodreads Book Reviews Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) (from UCSD Book Graph).

The project covers all steps of developing a recommendation system, including data loading, preprocessing, exploratory data analysis, model training (including hyperparameter tuning), model evaluation, and analysis of results.

## Project Structure and Steps

The project was carried out via a Jupyter Notebook comprising the following main steps:

1.  **Environment Setup & Configuration:**
    *   Importing necessary Python and PySpark libraries.
    *   Defining core project paths, a seed value (for reproducibility), and an error logging function.
    *   Collecting and displaying system information (OS, RAM, CPU).
    *   Initializing an optimized `SparkSession` and setting the checkpoint directory.

2.  **Dataset Acquisition & Initial Preparation:**
    *   Loading the `goodreads_reviews_dedup.json` dataset into a Spark DataFrame.
    *   Taking a 10% random sample (`sampled_df`) of the dataset for processing efficiency and caching it.

3.  **Data Preprocessing for Recommendation Model:**
    *   Selecting the `user_id`, `book_id`, and `rating` columns required for the ALS algorithm.
    *   Casting the `rating` column to `float` type.
    *   Cleaning missing values (nulls) (`als_df_cleaned`).
    *   Examining the rating distribution.
    *   Caching the final DataFrame (`df_for_als`).
    *   Demonstrating an alternative way to show rating distribution using the RDD API (`map`, `reduceByKey`).

4.  **Exploratory Data Analysis & Visualization:**
    *   Visualizing the rating distribution with a `countplot`.
    *   Identifying and visualizing the top 10 most rated books and top 10 most active users with `barplot`.

5.  **ALS Model Training & Hyperparameter Tuning:**
    *   Splitting the dataset into training (70%) and test (30%) sets.
    *   Converting string-type `user_id` and `book_id` to numerical indices using `StringIndexer` (fitted on training data and applied to both training and test sets).
    *   Checkpointing (`checkpoint()`) and caching the indexed training and test DataFrames for performance and fault tolerance.
    *   Implementing a hyperparameter tuning loop for the ALS model (`ranks`, `iterations`, `lambdas`):
        *   Loading progress from `RESULTS_JSON_PATH` to skip previously run parameter combinations.
        *   For each combination, training an ALS model, making predictions on the test set, and calculating RMSE (Root Mean Square Error).
        *   Saving results (parameters, metrics, duration) to `RESULTS_JSON_PATH`.
        *   Storing the model yielding the lowest RMSE as `best_model_obj_als`, saving the model to `BEST_MODEL_SAVE_PATH`, and its metadata to `BEST_MODEL_INFO_PATH`.
    *   Displaying the results of all tried models and details of the best model found.

6.  **Model Evaluation & Analysis:**
    *   Loading the best model from disk (if not already available in the session).
    *   Visualizing model performance against hyperparameters with line plots (RMSE vs. Rank, RMSE vs. Iterations).
    *   Performing a qualitative analysis by comparing the best model's predictions on the test set with actual ratings (converting indexed IDs back to original IDs using `IndexToString`).
    *   **Item-to-Item Recommendation (Cosine Similarity):**
        *   For a specific book (e.g., original ID "2767052"), calculating cosine similarity with other books using item factors from the ALS model and listing the top 5 most similar books.
    *   **User-Item Recommendation (Predicting Users for a Specific Item):**
        *   For the same target book, predicting potential user ratings by using the dot product of user factors and the item factor, and listing the top 10 users with the highest predicted ratings.

## File Structure (Project Outputs)

When the project is run, the following files and directories are created (or updated) under the `project/goodreads_models` directory:

*   `checkpoint_dir_als_project_v2/`: Contains checkpoint data used by Spark.
*   `project/goodreads_models/`:
    *   `best_model/`: Contains the saved best Spark ALS model.
    *   `best_model_info.json`: JSON file containing the parameters and metrics of the best model.
    *   `progress.json`: JSON file containing the results of all models tried during hyperparameter tuning.
    *   `errors.log`: Log file for errors encountered during execution.

## Setup and Execution

1.  **Requirements:**
    *   Java Development Kit (JDK 8 or 11)
    *   Python (3.8+)
    *   Apache Spark (3.x.x)
    *   PySpark (`pip install pyspark`)
    *   Pandas (`pip install pandas`)
    *   NumPy (`pip install numpy`)
    *   Matplotlib (`pip install matplotlib`)
    *   Seaborn (`pip install seaborn`)
    *   Jupyter Notebook/JupyterLab (`pip install notebook`)
    *   (For Windows) Hadoop binaries (winutils.exe)

2.  **Dataset:**
    *   Download the `goodreads_reviews_dedup.json` file into the main directory where the Jupyter Notebook is located.

3.  **Execution:**
    *   Open the Jupyter Notebook.
    *   Update the `YOUR_STUDENT_ID_LAST_4_DIGITS` variable (in Cell 1) with the last 4 digits of your student ID (this is used as a random seed).
    *   Run the cells sequentially. Model training (especially hyperparameter tuning) can take a significant amount of time. The `progress.json` file allows you to resume training from where you left off.

## Results

The notebook trains an ALS model by identifying the best hyperparameter set and uses this model to recommend both similar books and users who might be interested in a specific book. The RMSE value and parameters of the best model are stored in the notebook outputs and JSON files.

(For example, the notebook output shows that the best RMSE value obtained was 1.4816 with parameters Rank=200, Iterations=50, Lambda=0.1.)

## Contributors

*   Köksal Kerem TANIL (https://github.com/keremtanil)
*   Rıza KARAKAYA (https://github.com/RizaKarakaya)
*   Mustafa Cihan AYINDI (https://github.com/cihanayindi)
*   Yusuf TURAN (https://github.com/turan1609)

## References

*   [UCSD Book Graph Datasets](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html)
*   [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
*   [PySpark API Documentation](https://spark.apache.org/docs/latest/api/python/index.html)
*   [Alternating Least Squares (ALS) - Spark MLlib](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
