import os
from airflow.utils.email import send_email_smtp

recipient = os.getenv("EMAIL") # Set in docker-compose

# To send email with html content
def send_email(to, subject, html_content):
    send_email_smtp(
        to=to,
        subject=subject,
        html_content=html_content
    )

# Callback to send email on failure of any task in the dag
def send_failure_email(context):

    task_instance = context.get("task_instance")
    exception = context.get("exception")
    dag = context.get("dag")
    dag_run = context.get("dag_run")
    logical_date = context.get("logical_date")

    dag_id = getattr(dag, "dag_id", "unknown")
    task_id = getattr(task_instance, "task_id", "unknown")
    run_id = context.get("run_id", "unknown")
    execution_date = (
        logical_date
        or context.get("execution_date")
        or getattr(dag_run, "logical_date", None)
        or getattr(dag_run, "execution_date", None)
        or "unknown"
    )
    try_number = getattr(task_instance, "try_number", "unknown")
    log_url = getattr(task_instance, "log_url", "")
    error_reason = str(exception) if exception else context.get("reason", "No exception provided")

    subject = f"Airflow DAG Failed: {dag_id}"
    html_content = """
    <h2>Airflow DAG Failure</h2>
    <p><strong>DAG:</strong> {dag_id}</p>
    <p><strong>Task:</strong> {task_id}</p>
    <p><strong>Run ID:</strong> {run_id}</p>
    <p><strong>Execution Date:</strong> {execution_date}</p>
    <p><strong>Try Number:</strong> {try_number}</p>
    <p><strong>Error:</strong> {error_reason}</p>
    {log_link}
    """.format(
        dag_id=dag_id,
        task_id=task_id,
        run_id=run_id,
        execution_date=execution_date,
        try_number=try_number,
        error_reason=error_reason,
        log_link=(f'<p><strong>Log:</strong> <a href="{log_url}">View log</a></p>' if log_url else "")
    )

    send_email(recipient, subject, html_content)

# Callback utility to send email on success of the dag, with summary stats
def send_success_summary_email(context, stats):

    dag = context.get("dag")
    dag_run = context.get("dag_run")
    logical_date = context.get("logical_date")
    dag_id = getattr(dag, "dag_id", "unknown")
    run_id = context.get("run_id", "unknown")
    execution_date = (
        logical_date
        or context.get("execution_date")
        or getattr(dag_run, "logical_date", None)
        or getattr(dag_run, "execution_date", None)
        or "unknown"
    )

    stats_rows = "".join(
        f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in (stats or {}).items()
    ) or "<tr><td colspan=\"2\">No stats provided</td></tr>"

    subject = f"Airflow DAG Success: {dag_id}"
    html_content = """
    <h2>Airflow DAG Success</h2>
    <p><strong>DAG:</strong> {dag_id}</p>
    <p><strong>Run ID:</strong> {run_id}</p>
    <p><strong>Execution Date:</strong> {execution_date}</p>
    <h3>Summary Stats</h3>
    <table border="1" cellpadding="6" cellspacing="0">
        <thead>
            <tr><th>Metric</th><th>Value</th></tr>
        </thead>
        <tbody>
            {stats_rows}
        </tbody>
    </table>
    """.format(
        dag_id=dag_id,
        run_id=run_id,
        execution_date=execution_date,
        stats_rows=stats_rows
    )

    send_email(recipient, subject, html_content)



