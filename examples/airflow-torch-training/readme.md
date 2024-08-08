# Using Airflow with Runhouse 
The principal goal of using Runhouse alongside Airflow or other DAG systems is to restore interactive debuggability and fast iteration to developers. Packaging research code into production pipelines easily takes up half of machine learning engineers' time, and this is even true for sophisticated organizations. 

**Use Airflow for what Airflow is good for** 
* Scheduling 
* Ensuring reliable execution 
* Observability of runs 

The usage pattern for Runhouse with Airflow should be as follows:
* Write Python classes and functions using normal, ordinary coding best practices. Do not think about DAGs or DSLs at all, just write great code. 
* Send the code for remote execution with Runhouse, and figure out whether the code works, debugging it interactively. Runhouse lets you send the code in seconds, and streams logs back. You can work on remote as if it were local. 
* Once you are satisfied with your code, you can write the callables for an Airflow PythonOperator which is the minimal code required to operationalize already working Classes and Functions
* And then you can setup your DAG to be **minimal code beyond defining the order of calling steps** 
* And you can easily iterate further on your code, or test the pipeline end-to-end from local with no Airflow participation 
