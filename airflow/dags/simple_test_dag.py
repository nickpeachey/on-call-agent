from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'oncall-agent',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

def simple_task():
    """Simple test task"""
    print("Hello from Airflow!")
    return "success"

dag = DAG(
    'simple_test_dag',
    default_args=default_args,
    description='Simple test DAG',
    schedule_interval=None,
    is_paused_upon_creation=False,
    catchup=False,
    tags=['test', 'simple'],
)

task1 = PythonOperator(
    task_id='hello_task',
    python_callable=simple_task,
    dag=dag,
)

task2 = BashOperator(
    task_id='echo_task',
    bash_command='echo "This is a simple test task"',
    dag=dag,
)

task1 >> task2
