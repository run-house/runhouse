Data Layer
====================================
[PLACEHOLDER]

This guide introduces two of our core data primitives, `Table` and `Blob`.

Creating a Table
------------------
[PLACEHOLDER]

In order to use the table object, we will first need code.

Below we create a send which tokenizes a Huggingface dataset


.. code-block:: python

    import os
    import runhouse as rh
    from transformers import AutoTokenizer  # Hugging face's model library
    from datasets import load_dataset, load_metric  # Hugging face's data library

    # Derived from this Hugging Face tutorial:
    # https://huggingface.co/docs/transformers/training#train-in-native-pytorch

    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def tokenize_function(examples):
        return tokenizer(examples, padding="max_length", truncation=True)

    def tokenize_dataset(hf_dataset):
        return hf_dataset.map(tokenize_function,
                              input_columns=['text'],
                              batched=True,
                              num_proc=os.cpu_count())

Now we can create our send:

.. code-block:: python

    preproc = rh.Send(fn=tokenize_dataset,
                      hardware="rh_32_cpu")


.. note::
     For an example on how to implement a send check out our :ref:`Compute Layer`,
     or for a more formal explanation click :ref:`here <send>`.


Now that we have our code defined in our send, we can create our table object:

.. code-block:: python

    # Calling the microservice function with the Send callable
    preprocessed_yelp = preproc(dataset)

    # Saving down and naming the preprocessed data for reuse later or sharing
    rh.Table(data=preprocessed_yelp,
             name="yelp_bert_preprocessed")


Saving a Blob
------------------
[PLACEHOLDER]

.. code-block:: python

    preprocessed_data = rh.Table(name="yelp_bert_preprocessed")

    # ... do some training ...

    rh.Blob(data=pickle.dumps(model),
            name="yelp_model_output")





