import fastapi

app = fastapi.FastAPI()


@app.get("/summer/{a}")
def summer(a: int, b: int):
    return a + b


@app.post("/my/deeply/{arg1}/nested/endpoint/{arg2}")
async def my_deeply_nested_async_endpoint(arg1: str, arg2: int, arg3: float):
    return arg1, arg2, arg3


@app.get("/my/streaming/endpoint")
def my_streaming_endpoint():
    for i in range(10):
        yield i


@app.get("/my/endpoint/with/optional/body/params/and/header")
def my_endpoint_with_optional_body_params_and_header(
    a: int = fastapi.Body(None),
    b: int = fastapi.Body(None),
    c: int = fastapi.Header(None),
):
    return a, b, c


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
