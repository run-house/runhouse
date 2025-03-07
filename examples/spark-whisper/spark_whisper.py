import os
import tempfile
from pathlib import Path

import numpy as np

import pandas as pd
import runhouse as rh
import torch
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StringType, StructField, StructType


def init_spark(
    gpu_discovery_script=None, cores_per_executor=10, memory_per_executor="10g"
):
    # Initialize Spark with configuration based on available hardware
    spark_builder = SparkSession.builder.appName("Whisper Inference")

    gpu_count = torch.cuda.device_count()

    if gpu_count > 0 and gpu_discovery_script:
        spark_builder = (
            spark_builder.master(f"local[4]")
            .config("spark.executor.resource.gpu.discoveryScript", gpu_discovery_script)
            .config("spark.executor.cores", "10")
            .config("spark.executor.resource.gpu.amount", "1")
            .config("spark.task.resource.gpu.amount", "1")
            .config("spark.jars", "/home/ubuntu/rapids-4-spark_2.12-25.02.0.jar")
        )  # .config("spark.task.resource.gpu.amount", "1") \
        # .config("spark.rapids.sql.enabled","true") \
        # .config("spark.plugins", "com.nvidia.spark.SQLPlugin") \
        # .config("spark.executor.instances", str(gpu_count)) \
        # .config("spark.executor.cores", str(cores_per_executor)) \
        # .config("spark.executor.memory", "16g") \
        # .config("spark.driver.memory", "20g") \
        # .config("spark.executor.resource.gpu.vendor", "nvidia") \
        # .config("spark.default.parallelism", str(gpu_count)) \
        # .config("spark.cores.max", str(gpu_count * cores_per_executor))

    else:
        spark_builder = (
            spark_builder.master("local[*]")
            .config("spark.executor.cores", "12")
            .config("spark.executor.instances", "4")
            .config("spark.executor.memory", "40g")
            .config("spark.driver.memory", "20g")
        )

    spark = spark_builder.getOrCreate()
    spark.sparkContext.setLogLevel("DEBUG")

    # Toy check
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    def check_cuda():
        if torch.cuda.is_available():
            return f"CUDA devices: {torch.cuda.device_count()}"
        else:
            return "CUDA not available"

    # Register UDF
    cuda_udf = udf(check_cuda, StringType())

    # Example DataFrame to apply UDF to
    spark.range(1).withColumn("cuda_info", cuda_udf()).show()

    return spark


def transcribe(
    output_path,
    dataset_name,
    dataset_split=None,
    model_id="openai/whisper-large-v3-turbo",
    batch_size=10,
    gpu_discovery_script=None,
):

    spark = init_spark(gpu_discovery_script)

    result_schema = StructType(
        [StructField("id", StringType()), StructField("transcription", StringType())]
    )

    # Create a pandas UDF for batch inference with GPU acceleration if available
    @pandas_udf(result_schema)
    def transcribe_audio_batch(
        ids_series: pd.Series, audio_arrays: pd.Series, sampling_rates: pd.Series
    ) -> pd.DataFrame:
        import os

        import torch
        from pyspark import TaskContext
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        print("entering UDFs")

        task_context = TaskContext.get()
        print(task_context)
        if task_context and torch.cuda.is_available():
            gpu_resources = task_context.resources().get("gpu")
            if gpu_resources and len(gpu_resources.addresses) > 0:
                gpu_address = gpu_resources.addresses[0]
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_address
                device = f"cuda:0"
            else:
                device = "cpu"
        else:
            device = "cpu"

        device = (
            f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("Loading pipeline, on device ", device)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        try:
            model.to(device)
        except:
            print(task_context.resources())
            raise Exception(f"Error loading model on device {device}")
        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=batch_size,
            torch_dtype=torch.float16,
            device=device,
        )

        print("Process batches")
        results = []
        for idx, (sample_id, audio_array, sampling_rate) in enumerate(
            zip(ids_series, audio_arrays, sampling_rates)
        ):
            try:
                # Create input that the pipeline can process
                audio_input = {"array": audio_array, "sampling_rate": sampling_rate}
                result = pipe(audio_input)
                transcription = result["text"]
                results.append({"id": sample_id, "transcription": transcription})
            except Exception as e:
                results.append({"id": sample_id, "transcription": f"Error: {str(e)}"})

        return pd.DataFrame(results)

    print(f"Loading dataset: {dataset_name}, split: {dataset_split}")
    dataset = load_dataset(dataset_name, dataset_split)
    dataset = dataset[dataset_split].select(range(16))

    pd_df = pd.DataFrame(
        {
            "id": dataset["id"],
            "audio_array": [audio["array"] for audio in dataset["audio"]],
            "sampling_rate": [audio["sampling_rate"] for audio in dataset["audio"]],
            "original_text": dataset["text"],
        }
    )
    pd_df["audio_array"] = pd_df["audio_array"].apply(lambda x: x.tolist())

    print("Creating Spark DataFrame")
    spark_df = spark.createDataFrame(pd_df)
    num_partitions = max(1, len(pd_df) // batch_size)
    print(f"Repartitioning to {num_partitions} partitions")
    spark_df = spark_df.repartition(num_partitions)

    # Apply the UDF for parallel transcription
    print("Starting distributed transcription")
    result_df = spark_df.withColumn(
        "result", transcribe_audio_batch("id", "audio_array", "sampling_rate")
    ).select("id", "original_text", col("result.transcription").alias("transcription"))

    # result_df.show(5, truncate=False)

    result_df.write.parquet(output_path, mode="overwrite")
    print("Distributed transcription complete")
    spark.stop()

    return f"Transcription complete. Results saved to {output_path}"


def setup_commands(cluster):
    cluster.rsync(
        "/Users/paulyang/runhouse/examples/spark-whisper/gpu-discovery.sh", "~/"
    )
    cluster.run_bash_over_ssh(["chmod +x gpu-discovery.sh"])

    # Install JRE
    cluster.run_bash_over_ssh(
        [
            "sudo apt-get update && apt-get install -y wget",
            "wget https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz",
            "tar -xvf openjdk-11.0.2_linux-x64_bin.tar.gz",
            'grep -q "export JAVA_HOME=" ~/.bashrc || echo "export JAVA_HOME=~/jdk-11.0.2" >> ~/.bashrc',
            'grep -q "export PATH=$JAVA_HOME/bin:$PATH" ~/.bashrc || echo "export PATH=$JAVA_HOME/bin:$PATH" >> ~/.bashrc',
        ]
    )

    # Install Spark
    cluster.run_bash_over_ssh(
        [
            "wget https://dlcdn.apache.org/spark/spark-3.4.4/spark-3.4.4-bin-hadoop3.tgz",
            "tar --overwrite -xvzf spark-3.4.4-bin-hadoop3.tgz",
            "grep -q 'export SPARK_HOME=' ~/.bashrc || echo 'export SPARK_HOME=~/spark-3.4.4-bin-hadoop3' >> ~/.bashrc",
            "grep -q 'export PATH=\$SPARK_HOME/bin:\$SPARK_HOME/sbin:\$PATH' ~/.bashrc || echo 'export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH' >> ~/.bashrc",
            "grep -q 'export PYSPARK_PYTHON=python3' ~/.bashrc || echo 'export PYSPARK_PYTHON=python3' >> ~/.bashrc",
        ]
    )

    # Install RAPIDS
    cluster.run_bash_over_ssh(
        [
            "wget https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/25.02.0/rapids-4-spark_2.12-25.02.0.jar",
            "sudo apt install nvidia-cuda-toolkit",
        ]
    )

    # Set CUDA_HOME and LD_LIBRARY_PATH in ~/.bashrc
    cluster.run_bash_over_ssh(
        [
            "grep -q 'export CUDA_HOME=/usr/local/cuda' ~/.bashrc || echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc",
            "grep -q 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' ~/.bashrc || echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc",
        ]
    )


if __name__ == "__main__":

    img = rh.Image().install_packages(
        [
            "pyspark",
            "whisper",
            "pandas",
            "torch",
            "transformers",
            "datasets",
            "librosa",
            "soundfile",
        ],
    )

    num_gpus_per_node = 4

    cluster = rh.compute(
        name=f"rh-L4x{num_gpus_per_node}",
        instance_type=f"L4:{num_gpus_per_node}",
        provider="aws",
        image=img,
        use_spot=False,
        autostop_mins=1200,
    ).up_if_not()  # Requires access to a cloud account with the necessary permissions to launch compute.
    cluster.restart_server()
    # cluster.teardown()

    # setup_commands(cluster)

    # spark-shell \
    #   --jars /home/ubuntu/rapids-4-spark_2.12-25.02.0.jar \
    #   --conf spark.plugins=com.nvidia.spark.SQLPlugin \
    #   --conf spark.rapids.sql.enabled=true \
    #   --conf spark.executor.resource.gpu.amount=1 \
    #   --conf spark.task.resource.gpu.amount=1

    remote_transcribe = rh.function(transcribe).to(cluster)

    remote_transcribe(
        dataset_name="MLCommons/peoples_speech",
        dataset_split="test",
        output_path="./peoples_speech_transcriptions",
        model_id="openai/whisper-large-v3-turbo",
        batch_size=16,
        gpu_discovery_script="/home/ubuntu/gpu-discovery.sh",
    )
