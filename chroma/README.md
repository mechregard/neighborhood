# Evaluation of chromadb and clip embeddings
Simple CLI and demo webapp (based on gradio)

## CLI
Create embeddings and populate database, run simple queries with CLI:
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
Example to prepare the database index using the different clip embedders, run the following. When the command
exits, the database will be persisted within the db/ directory:
```
python client prepare --index=image_clip
```
and
```
python client prepare --index=image_fclip
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