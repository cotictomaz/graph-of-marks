# Standard image for the SLURM cluster's RTX 3090 / Titan Xp nodes
# (faretra, moro43, moro232, deeplearn2). For the RTX 5090 node, build
# build/Dockerfile.rtx5090 instead (see that file and README.md for details).
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="disi-unibo-nlp"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory (do not rename: standardized convention on the cluster)
WORKDIR /workspace
ENV APP_PATH=/workspace

# Install general-purpose + OpenCV runtime dependencies
RUN apt-get update -y && \
    apt-get install -y curl \
                        git \
                        bash \
                        nano \
                        python3.11 \
                        python3-pip \
                        libgl1 \
                        libglib2.0-0 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gdown

COPY build/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# detectron2 is not distributed on PyPI; build it against the torch version
# installed above (--no-deps avoids pulling a second, unpinned torch).
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
        "git+https://github.com/facebookresearch/detectron2.git@main"

# Model assets used by gom.relations (spaCy pipeline, NLTK wordnet)
RUN python3 -m spacy download en_core_web_md && \
    python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Register the gom package + console scripts (gom-preprocess, gom-vqa) as an
# editable install. The project directory is normally bind-mounted over
# /workspace at `docker run` time (see run_docker.sh); since the mount keeps
# the same src/ layout, this editable link keeps working against the live
# code without needing a rebuild on every code change.
COPY setup.py pyproject.toml MANIFEST.in README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog

CMD ["/bin/bash"]
