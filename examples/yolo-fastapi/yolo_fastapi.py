import io
import json

import requests
import runhouse as rh
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image


class YOLOv3:
    def __init__(self):
        self.model = None
        self.device = None

    def load_model_torchhub(self, repo="WongKinYiu/yolov7", model_id="yolov7"):
        self.model = torch.hub.load(
            repo, model_id, force_reload=True, trust_repo=True, pretrained=True
        )
        print("Loaded model")

    def process_images(self, img_urls, img_size=640):
        imgs = []
        for img_url in img_urls:
            try:
                imgs.append(self.process_image(img_url, img_size))
            except:
                print(f"Error processing image: {img_url}")

        return imgs

    def process_image(self, img_url, img_size=640):
        # Load the image
        image = Image.open(io.BytesIO(requests.get(img_url).content))
        # Validate image type
        if image.format not in ["JPEG", "PNG"]:
            raise ValueError("Unsupported image format. Only JPEG and PNG are allowed.")
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Resize to the desired size
        return image.resize((img_size, img_size))

    def predict(self, images):
        images = self.process_images(images)

        if self.model is None:
            print("Did not yet load model, loading model")
            self.load_model_torchhub()

        result = self.model(images)
        result.print()
        return result.pandas().xyxy


def bring_up_remote_inference_service():
    img = rh.Image(name="rh-yolov3").pip_install(
        [
            "torch",
            "torchvision",
            "Pillow",
            "numpy",
            "opencv-python",
            "matplotlib",
            "seaborn",
            "scipy",
        ]
    )
    cluster = rh.compute(
        name="rh-yolov3", image=img, instance_type="A10G:1", provider="aws"
    )

    if not cluster.is_up():
        cluster = cluster.up_if_not()
        cluster.run_bash(
            "curl -O -L https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
        )
        remote_YOLO = rh.cls(YOLOv3).to(cluster, name="yolo")
        YOLO = remote_YOLO(name="yolo_model")

    else:
        try:
            YOLO = cluster.get("yolo_model", remote=True)
        except:
            remote_YOLO = rh.cls(YOLOv3).to(cluster, name="yolo")
            YOLO = remote_YOLO(name="yolo_model")

    return YOLO


# # FastAPI Server

app = FastAPI()
# ## FastAPI Inference Endpoint
# Example call
# curl -X POST "http://127.0.0.1:1357/predict/" \
# -H "Content-Type: application/json" \
# -d '{
#     "urls": [
#         "https://cdn.sanity.io/images/cefqallt/production/fdf8f21bf27ba9f54ad55a8e552629bbe6f8444f-1024x1024.jpg",
#         "https://cdn.sanity.io/images/cefqallt/production/3f164f15f123db4a43f46cb9c08a94cd3733a408-1024x1024.png"
#     ]
# }'


@app.post("/predict/")
async def predict(request: Request):
    try:
        body = await request.json()
        urls = body.get("urls", [])

        # Bring up the cluster
        YOLO = bring_up_remote_inference_service()

        # Run inference on remote
        results = YOLO.predict(urls)

        result_dicts = [df.to_dict(orient="records") for df in results]
        return JSONResponse(content=json.dumps(result_dicts))

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
def read_root():
    return {"message": "YOLOv3 FastAPI server is running!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=1357)
