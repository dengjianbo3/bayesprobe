FROM python:3.12.10-slim-bookworm@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
      build-essential \
      libgmp-dev \
      libmpc-dev \
      libmpfr-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels \
      galois==0.4.4 \
      gmpy2==2.2.1 \
      mpmath==1.3.0 \
      networkx==3.4.2 \
      numpy==2.1.3 \
      scipy==1.15.2 \
      sympy==1.13.3

FROM python:3.12.10-slim-bookworm@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=builder /wheels /wheels

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgmp10 libmpc3 libmpfr6 \
    && python -m pip install --no-index /wheels/*.whl \
    && rm -rf /wheels /var/lib/apt/lists/* \
    && groupadd --gid 65532 sandbox \
    && useradd --uid 65532 --gid 65532 --no-create-home --shell /usr/sbin/nologin sandbox

WORKDIR /tmp
USER 65532:65532

CMD ["python", "-s", "-"]
