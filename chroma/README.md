# Evaluation of chromadb and clip embeddings
Simple CLI and demo webapp (based on gradio)

## CLI
Ingest datasets, create embeddings and populate database, run simple queries with CLI.
The indexes within the vector database are defined by the source dataset and the encoder
model used.
```
$ python client.py --help
Usage: client.py [OPTIONS] COMMAND [ARGS]...

  test out chromadb

Options:
  --help  Show this message and exit.

Commands:
  prepare  Build collection using given name
  search   Search for images matching given image path
```
#### Ingest
Create a HF dataset with images in "images" column (not URIs) and serialize into the datasets/ directory.
The source can be a json blob (type=json) or an online HF datset (type=hub).
To support query retrieval, all source images are persisted locally and accessible via ID.

Two initial datasets ingested this way:
```
python client.py ingest --src="scraped/staud.json" --type=json --target=staud
```
and
```
python client.py ingest --src="ashraq/fashion-product-images-small" --type=hub --target="fashion-product-images-small"
```
#### Prepare DB Index
Create an index from a given dataset and encoder method. The index name is a combination of these names.
For the two initial encoding models and two initial datasets, there are 4 indexes:
```
# create index staud_image_clipb
python client.py prepare --dataset=staud --encoder=image_clipb
```
and
```
# create index staud_image_clipl
python client.py prepare --dataset=staud --encoder=image_clipl
```


Do run a simple query against either index using a path to an image:
```
$ python client.py search --help
Usage: client.py search [OPTIONS]

  Search for images matching given image path

Options:
  --index TEXT     Type {image_clip|image_fclip}
  --image TEXT     Pathname to image file (default None)
  --category TEXT  Filter category (default None)
  --help           Show this message and exit.
```

## Docker
When running the container, expose the port:
```
docker run -p 9090:9090 --rm -d CONTAINER_ID
```
then open browser to `localhost:9090`