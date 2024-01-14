ARG PYTHON_VERSION_LAMBDA
ARG LAMBDA_FILES
ARG LAMBDA_HANDLER
ARG REQS

FROM public.ecr.aws/lambda/python:$PYTHON_VERSION_LAMBDA

# If exist, copy requirements.txt file and install it. Otherwise, install list of reqs.
RUN if [ $REQS = "requirements.txt" ] ; then \
    echo "requirements.txt exists, copying it..."; \
    cp requirements.txt $LAMBDA_TASK_ROOT ;
    pip install -r requirements.txt ; \
    else \
        pip install $REQS ;
    fi


# Copy function code
# COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY $LAMBDA_FILES ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
# CMD [ "lambda_function.handler" ]
CMD [ "$LAMBDA_HANDLER" ]
