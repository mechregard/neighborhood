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
  ingest   Build dataset from file
  prepare  Build index using given dataset, encoder
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
Create a vector db index from a given dataset and encoder method. The index name is a combination of these names.
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

#### Search
Do run a simple query against an index using a path to an image. Optionally add category to filter and 
boolean flag to display actual image results.
```
$ python client.py search --help
Usage: client.py search [OPTIONS]

  Search for images matching given image path

Options:
  --index TEXT     in form: dataset_encoder
  --image TEXT     Pathname to query image file (default None)
  --category TEXT  Filter category (default None)
  --show BOOLEAN   True to display
  --help           Show this message and exit.
```
An example of searching the staud clib-b index for similar images to a demo image:
```
# create index staud_image_clipl
python client.py search --index=staud_image_clipb --image=images/demo.jpeg --show=True
```

## Demo
Run demo locally (requires python 3.9.6 and poetry 1.5.1)
```
poetry install
poetry run python demo.py
```
Then point browser to `localhost:9090`

## Docker
When running the container, expose the port:
```
docker run -p 9090:9090 --rm -d CONTAINER_ID
```
then open browser to `localhost:9090`

## Data Directories used
The following directories are used:
```
scraped/ contains json datafiles, with URIs to images. 
datasets/ contains huggingface datasets, created by ingest command from either a scraped src or HF hub dataset
images/ persisted src images by ID to simplify how demo displays search results
db/ chromaDB persisted indexes and a json mapping of index name to encoder
```