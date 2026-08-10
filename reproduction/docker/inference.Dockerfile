FROM vllm/vllm-openai:v0.10.2

RUN python3 -m pip install --no-cache-dir --upgrade \
    transformers==4.56.2 pillow==12.0.0 numpy==2.2.6 pyyaml==6.0.2

WORKDIR /workdir
COPY pyproject.toml README.md LICENSE /workdir/
COPY src /workdir/src
COPY reproduction /workdir/reproduction
RUN python3 -m pip install --no-deps -e /workdir

LABEL org.graph-of-marks.vllm="0.10.2" \
      org.graph-of-marks.transformers="4.56.2"

ENTRYPOINT []
