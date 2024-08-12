# RAG App with Vector Embedding and LLM Generation

This example defines a retrieval augmented generation (RAG) app that references text from websites
to enrich the response from an LLM. Depending on the URLs you use to populate the vector database,
you'll be able to answer questions more intelligently with relevant context.

You could, for example, pass in the URL of a new startup and then ask the app to answer questions
about that startup using information on their site. Or pass in several specialized gardening websites and
ask nuanced questions about horticulture.

## Example Overview
Deploy a FastAPI app that is able to create and store embeddings from text on public website URLs,
and generate answers to questions using related context from stored websites and an open source LLM.

#### Indexing:
- **Send a list of URL paths** to the application via a POST endpoint
- Use [LangChain](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/)
  to **parse text from URLs**
- **Create vector embeddings from split text** with [Sentence Transformers (SBERT)](https://sbert.net/index.html)
- Store embeddings in a **vector database** ([LanceDB](https://lancedb.com/))

#### Retrieval and generation:
- **Send question text** via a GET endpoint on the FastAPI application
- Create a vector embedding from the text and **retrieve related docs** from database
- **Construct an LLM prompt** from documents and original question
- Generate a response using an **LLM (Llama 3)**
- **Output response** with source URLs and question input

![Graphic displaying the steps of indexing data and the retrieval and generation process](https://runhouse-tutorials.s3.amazonaws.com/indexing-retrieval-generation.png)

Note: Some of the steps in this example could also be accomplished with platforms like OpenAI and
tools such as LangChain, but we break out the components explicitly to fully illustrate each step and make the
example easily adaptible to other use cases. Swap out components as you see fit!

### What does Runhouse enable?
Runhouse allows you to turn complex operations such as preprocessing and inference into independent services.
Servicifying with Runhouse enables:
- **Decoupling**: AI/ML or compute-heavy heavy tasks can be separated from your main application.
  This keeps the FastAPI app light and allows each service to scale independently.
- **Multi-cloud**: Host Runhouse modules on remote machines from any cloud provider (even on the same GPU).
  Take advantage of unused cloud credits and avoid platform dependence.
- **Sharing**: Running on GPUs can be expensive, especially if they aren't fully utilized. By sharing services within
  your organization, you can substantially cut costs.

### Why FastAPI?
We chose FastAPI as our platform because of its popularity and simplicity. However, we could
easily use any other *Python-based* platform with Runhouse. Streamlit, Flask, `<your_favorite>`, we got you!

## Setup credentials and dependencies

To ensure that Runhouse is able to manage deploying services to your cloud provider (AWS in this case)
you may need to follow initial setup steps. Please visit the AWS section of
our [Installation Guide](https://www.run.house/docs/installation)

Additionally, we'll be downloading the Llama 3 model from Hugging Face, so we need to set up our Hugging Face token:
```shell
$ export HF_TOKEN=<your huggingface token>
```

Make sure to sign the waiver on the [Hugging Face model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
so that you can access it.

## Run the FastAPI App Locally
Use the following command to run the app from your terminal:

```shell
$ fastapi run app/main.py
```

After a few minutes, you can navigate to `http://127.0.0.1/health` to check that your application is running.
This may take a while due to initialization logic in the lifespan.

You'll see something like:
```json
{ "status": "healthy" }
```

To debug the application, you may prefer running `fastapi dev`. This will trigger
automatic re-deployments from any changes to your code. Be sure to set `DEBUG` to `True` to
override instances of the embedding and LLM services with updated versions.

### Example cURL Command to Add Embeddings
To populate the LanceDB database with vector embeddings for use in the RAG app, you can send a HTTP request
to the `/embeddings` POST endpoint. Let's say you have a question about bears. You could send a cURL
command with a list of URLs including essential bear information:

```shell
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"paths":["https://www.nps.gov/yell/planyourvisit/safety.htm", "https://en.wikipedia.org/wiki/Adventures_of_the_Gummi_Bears"]}' \
  http://127.0.0.1:8000/embeddings
```

Alternatively, we recommend a tool like [Postman](https://www.postman.com/) to test HTTP APIs.

### Test the Generation Endpoint
Open your browser and send a prompt to your locally running RAG app by appending your question
to the URL as a query param, e.g. `?text=Does%20yellowstone%20have%20gummi%20bears%3F`

```text
"http://127.0.0.1/generate?text=Does%20yellowstone%20have%20gummi%20bears%3F"
```
#
The `LlamaModel` will need to load on the initial call and may take a few minutes to generate a
response. Subsequent calls will generally take less than a second.

Example output:

```json
{
  "question": "Does yellowstone have gummi bears?",
  "response": [
    " No, Yellowstone is bear country, not gummi bear country. Thanks for asking! "
  ],
  "sources": [
    "https://www.nps.gov/yell/planyourvisit/safety.htm",
    "https://en.wikipedia.org/wiki/Adventures_of_the_Gummi_Bears"
  ]
}
```

## Deploying to Production
There are any methods to deploy a FastAPI application to a production environment. With some modifications
to the logic of the app, setting `DEBUG` to `False`, and deploying to a public IP, this example
could easily serve as the backend to a RAG app.

We won't go into depth on a specific method, but here are a few things to consider:
- **Cloud credentials**: Runhouse uses [SkyPilot](https://github.com/skypilot-org/skypilot) to provision remote
  machines on various cloud providers. You'll need to ensure that you have the appropriate permissions available
  on your production environment.
- **ENV Variables**: This example passes your Hugging Face token to a Runhouse `env` to grant permissions to use
  Llama 3. Make sure you handle this on your server as well.
- **Cluster lifecycle**: Running GPUs can be costly. `sky` commands make it easy to manage remote clusters locally
  but you may also want to monitor your cloud provider to avoid unused GPUs running up a bill.

If you're running into any problems using Runhouse in production, please reach out to our team at
[team@run.house](mailto:team@run.house). We'd be happy to set up a time to help you debug live.
Additionally, you can chat with us directly on [Discord](https://discord.com/invite/RnhB6589Hs).
