from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging

# Default arguments for the DAG
default_args = {
    'owner': 'oncall-agent',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def test_restart_task(**context):
    """Simulate a restart task that the on-call agent would trigger"""
    logging.info("🔄 Airflow DAG triggered by on-call agent!")
    logging.info(f"Task instance: {context['task_instance']}")
    logging.info(f"Execution date: {context['execution_date']}")
    
    # Simulate some work
    import time
    time.sleep(2)
    
    logging.info("✅ Simulated ETL restart completed successfully")
    return "restart_completed"

# Create the DAG
dag = DAG(
    'test_dag_for_oncall_agent',
    default_args=default_args,
    description='Test DAG for on-call agent integration',
    schedule_interval=None,  # Manual trigger only
    is_paused_upon_creation=False,  # Start unpaused
    catchup=False,
    tags=['oncall', 'test', 'etl'],
)

# Define the task
restart_task = PythonOperator(
    task_id='simulate_etl_restart',
    python_callable=test_restart_task,
    dag=dag,
)

# Set task dependencies (single task in this case)
restart_task
