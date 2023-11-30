def num_cpus():
    import multiprocessing
    return f"Num cpus: {multiprocessing.cpu_count()}"