# Using Airflow with Runhouse 
The principal goal of using Runhouse alongside Airflow or other DAG systems is to restore interactive debuggability and fast iteration to developers. Packaging research code into production pipelines easily takes up half of machine learning engineers' time, and this is even true for sophisticated organizations. 

**Use Airflow for what Airflow is good for** 
* Scheduling 
* Ensuring reliable execution 
* Observability of runs 

The usage pattern for Runhouse with Airflow should be as follows:
* Write Python classes and functions using normal, ordinary coding best practices. Do not think about DAGs or DSLs at all, just write great code. 
* Send the code for remote execution with Runhouse, and figure out whether the code works, debugging it interactively. Runhouse lets you send the code in seconds, and streams logs back. You can work on remote as if it were local. 
* Once you are satisfied with your code, you can write the callables for an Airflow PythonOperator. The code that is actually in the Airflow DAG is the **minimal code** to call out to already working Classes and Functions, defining the order of the steps (or you can even have a one-step Airflow DAG, making Airflow purely for scheduling and observability)
* And you can easily iterate further on your code, or test the pipeline end-to-end from local with no Airflow participation 


**Examples**
* **torch_example_for_airflow.py:** Normally written Python code with no DSL, defining a simple neural network. 
* **airflow_example_torch_train.py:** The Airflow DAG, which simply orchestrates the pipeline. 
* **local_run_of_callables.py:** An example of how Runhouse lets you test your functions and Airflow callables from local, since it's all happening on "remote" execution. You can update code, and experiment with calling just that step. 
