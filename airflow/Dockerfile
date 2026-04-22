# Latest airflow image
FROM apache/airflow:3.2.1-python3.11

# required root privilege for creating directories and setting permissions for airflow user
USER root
RUN mkdir -p /opt/airflow/baseline_data /opt/airflow/feedback_data && \
    chown -R airflow:0 /opt/airflow/baseline_data /opt/airflow/feedback_data && \
    chmod -R g+rwx /opt/airflow/baseline_data /opt/airflow/feedback_data

# Switch to normal airflow user
USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY dags/ /opt/airflow/dags/
COPY plugins/ /opt/airflow/plugins/
