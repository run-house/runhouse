import runhouse as rh 

import ray 
import ray.data 
from dlrm_training import DLRM, read_preprocessed_dlrm
import torch 

# DLRM model for inference as required by Ray Data, that reads the model from S3 
class DLRMInferenceModel: 
    def __init__(self, unique_users, unique_movies, embeddings_dim, model_s3_bucket, model_s3_key):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DLRM(
            num_users=unique_users,
            num_items=unique_movies,
            embedding_dim=embeddings_dim,
        )

        self.model.load_model(s3_bucket=model_s3_bucket, s3_key=model_s3_key)
        self.model.to(self.device)
    
    def __call__(self, batch):
        users = torch.tensor(batch["userId"], device=self.device)
        movies = torch.tensor(batch["movieId"], device=self.device)
    
        prediction = self.model(users, movies).cpu().detach().numpy()
    
        return {"userId": batch["userId"], "movieId": batch["movieId"], "prediction": prediction}
    
# Function to be called remotely on GPU cluster to do the inference 
def inference_dlrm(num_gpus, num_nodes, model_s3_bucket, model_s3_key, dataset_s3_path, write_s3_path):
    unique_users = 330975 
    unique_movies = 86000 
    embeddings_dim = 64
    
    dlrm_model = DLRMInferenceModel(unique_users, unique_movies, embeddings_dim, model_s3_bucket, model_s3_key)
    
    ds = read_preprocessed_dlrm(dataset_s3_path)
    
    predictions = ds.map_batches(
        dlrm_model,
        num_gpus=num_gpus,
        batch_size=128,
        concurrency=num_nodes,
    )
    
    predictions.show(limit=1)

    # Write predictions to s3
    predictions.write_parquet(write_s3_path)

# Runhouse cluster setup and function call
if __name__ == "__main__":
    gpus_per_node = 1
    num_nodes = 2

    gpu_cluster = (
        rh.cluster(
            name=f"rh-{num_nodes}x{gpus_per_node}GPU",
            instance_type=f"A10G:{gpus_per_node}",
            num_nodes=num_nodes,
            provider="aws",
            autostop_minutes = 45,
            default_env=rh.env(
                name="pytorch_env",
                reqs=[
                    "torch==2.5.1",
                    "datasets",
                    "boto3",
                    "awscli",
                    "ray[data,train]"
                ],
            ),
        )
        .up_if_not()
        .save()
    )
    #gpu_cluster.restart_server()
    gpu_cluster.sync_secrets(["aws"])
    
    remote_inference = rh.function(inference_dlrm).to(gpu_cluster, name = "inference_dlrm").distribute('ray')
    remote_inference(num_gpus = gpus_per_node, 
                     num_nodes = num_nodes,
                     model_s3_bucket="rh-demo-external",
                     model_s3_key="dlrm-training-example/checkpoints/dlrm_model.pth",
                     dataset_s3_path = "s3://rh-demo-external/dlrm-training-example/preprocessed_data/test/", 
                     write_s3_path = "s3://rh-demo-external/dlrm-training-example/predictions/")
    
    gpu_cluster.teardown()