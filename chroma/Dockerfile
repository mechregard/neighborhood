FROM acidrain/python-poetry:3.9-slim-1.5.1

RUN useradd -d /home/ssdemo -s /bin/bash ssdemo && \
    mkdir /home/ssdemo && chown -R ssdemo:ssdemo /home/ssdemo && \
    echo 'export PATH=$HOME/.local/bin/:$PATH' >> /home/ssdemo/.bashrc

RUN apt-get update && apt-get install -y build-essential  gcc git
RUN pip install git+https://github.com/nmslib/hnswlib.git
RUN poetry config cache-dir /home/ssdemo/poetry-cache && mkdir /home/ssdemo/poetry-cache

RUN mkdir -p /home/ssdemo/chroma
COPY . /home/ssdemo/chroma/

# Installs dev dependencies (optionally export HNSWLIB_NO_NATIVE=1)
RUN cd /home/ssdemo/chroma && poetry install -vvv

WORKDIR /home/ssdemo/chroma
CMD ["/home/ssdemo/chroma/boot.sh"]
