FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install .

COPY . .

CMD ["flwr", "run", "."]