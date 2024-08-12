#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash import BashOperator

user = "user"

# Default arguments for the DAG
default_args = {
    "owner": user,
    "depends_on_past": False,
    "start_date": datetime(2024, 8, 23),  # Starting on August 23, 2024
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
with DAG(
    "four_month_interval_model_refresh",
    default_args=default_args,
    description="A DAG to run Jupyter notebooks and export metrics every 4 months on specific dates",
    schedule_interval="0 0 23 */4 *",  # Runs at midnight on the 23rd of every 4th month
    catchup=False,
) as dag:

    start = DummyOperator(task_id="start")

    # Task to open Jupyter notebook server
    start_jupyter = BashOperator(
        task_id="start_jupyter",
        bash_command='jupyter notebook --no-browser --port=8888 --NotebookApp.token="" &',
    )

    # Task to run DataProcessing.ipynb, update file path as necessary for repo directory
    run_data_processing = BashOperator(
        task_id="run_data_processing",
        bash_command="jupyter nbconvert --to notebook --execute /Pharma-Drug-Surveillance/CodeLibrary/DataProcessing.ipynb --output /path/to/your/repository/DataProcessing_output.ipynb",
    )

    # Task to run Modeling-with-ImputedPrices.ipynb, update file path as necessary for repo directory
    run_modeling = BashOperator(
        task_id="run_modeling",
        bash_command="jupyter nbconvert --to notebook --execute /Pharma-Drug-Surveillance/CodeLibrary/Modeling-with-ImputedPrices.ipynb --output /path/to/your/repository/Modeling_with_ImputedPrices_output.ipynb",
    )

    # Task to export metrics to the specified location, update file path as necessary for repo directory
    export_metrics = BashOperator(
        task_id="export_metrics",
        bash_command="cp /Pharma-Drug-Surveillance/DataLibrary/EvaluationMetrics/aggregated_metrics_withpricesImputed.csv C://Users/{user}/Downloads/model_refresh_metrics.csv",
    )

    # Dummy task to signify the end of the process
    end = DummyOperator(task_id="end")

    # Set the task dependencies
    (
        start
        >> start_jupyter
        >> run_data_processing
        >> run_modeling
        >> export_metrics
        >> end
    )
