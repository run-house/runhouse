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

### What does Kubetorch enable?
Kubetorch allows you to turn complex operations such as preprocessing and inference into independent services.
- **Decoupling**: AI/ML or compute-heavy heavy tasks can be separated from your main application.
  This keeps the FastAPI app light and allows each service to scale independently.
- **Multi-cloud**: Host Kubetorch modules on remote machines from any cloud provider (even on the same GPU).
  Take advantage of unused cloud credits and avoid platform dependence.
- **Sharing**: Running on GPUs can be expensive, especially if they aren't fully utilized. By sharing services within
  your organization, you can substantially cut costs.

### Why FastAPI?
We chose FastAPI as our platform because of its popularity and simplicity. However, we could
easily use any other *Python-based* platform with Kubetorch. Streamlit, Flask, `<your_favorite>`, we got you!

## Setup credentials and dependencies
We'll be downloading the Llama 3 model from Hugging Face, so we need to set up our Hugging Face token:
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
