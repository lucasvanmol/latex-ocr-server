FROM python:3.12-slim

RUN apt-get update 
RUN python -m pip install hatch

RUN mkdir /app
WORKDIR /app


COPY pyproject.toml /app


# Install dependencies
RUN hatch dep show requirements > /app/requirements.txt && \
    python -m pip install -r /app/requirements.txt

VOLUME [ "/app/model_cache" ]

# Install APP
COPY . /app

RUN python -m pip install -e .

CMD ["python", "-m", "latex_ocr_server", "start", "--cache_dir", "/app/model_cache", "--no-download"]