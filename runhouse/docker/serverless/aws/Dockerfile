ARG PYTHON_VERSION_LAMBDA

FROM public.ecr.aws/lambda/python:$PYTHON_VERSION_LAMBDA

ARG LAMBDA_HANDLER
ARG REQS
ARG IS_REQS_TXT


# If exist, copy requirements.txt file and install it. Otherwise, install list of reqs.
RUN if [ "$IS_REQS_TXT" = "True" ] ; then \
    echo "requirements.txt exists, copying it..."; \
    cp $REQS ${LAMBDA_TASK_ROOT}; \
    pip install -r requirements.txt; \
else \
    pip install $REQS; fi
