from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import click
import json
import pandas as pd
from PIL import Image as img
from PIL.Image import Image
import torch
import datasets
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class TryChroma:
    DATASET_NAME = "ashraq/fashion-product-images-small"
    TYPE_CLIPB = "image_clipb"
    TYPE_CLIPL = "image_clipl"
    MAX_RESULTS = 5

    def __init__(self):
        """
        Collection tied to embedder. Preload embedders here
        """
        logging.basicConfig(format='%(levelname)s %(asctime)s %(module)s: %(message)s',
                          datefmt='%Y-%m-%d,%H:%M:%S',
                          level=logging.INFO)
        self._chroma_client = chromadb.Client(Settings(
                                    chroma_db_impl="duckdb+parquet",
                                    persist_directory="./db"
                                ))
        dd = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._collection_encoder = {
            TryChroma.TYPE_CLIPB: SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=dd),
            TryChroma.TYPE_CLIPL: SentenceTransformer('sentence-transformers/clip-ViT-L-14', device=dd)
        }

    @staticmethod
    def ingest(name: str, type: str, target: str):
        """
        ingest data into local hf dataset
        """
        target_path = Path("datasets", target)
        if type == "hub":
            logging.info(f"Ingesting {name=} {type=} to {target}")
            dataset = load_dataset(
                name,
                split="train"
            )
            dataset.save_to_disk(target_path)
        elif type == "json":
            logging.info(f"Ingesting {name=} {type=} to {target}")
            with open(name) as f:
                objects = json.load(f)
            dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=objects)).cast_column("image", datasets.Image())
            dataset.save_to_disk(target_path)
        else:
            logging.error(f"Ingest skipped unknown {type}")

    def prepare(self, dataset_name: str, encoder_name: str) -> None:
        """
        Prepare for usage by indexing images from a dataset,
        using embedder based in index name, and saving images by id
        for search results.
        :param dataset_name:
        :param encoder_name:
        :return:
        """
        logging.info(f"Prepare loading {dataset_name} encoder {encoder_name}")
        index_name = dataset_name+"_"+encoder_name
        images, metadata = self.load_dataset(dataset_name)
        embeddings = self.embeddings(encoder_name, images)
        self.build_index(index_name, embeddings, metadata)
        self._save_images(index_name, images, metadata)
        TryChroma._update_encoder_mapping(index_name, encoder_name)

    def search(self,
               query_image: Image,
               constraints: Dict[str, str],
               index_name: Optional[str] = "staud_image_clipb") -> List[Dict[str,Any]]:
        """
        Answer back a list of metadata search results from the given Image
        and query constraints
        :param query_image:
        :param constraints:
        :param index_name:
        :param encoder:
        :return:
        """
        encoder = TryChroma._inflate_encoder_mapping()[index_name]
        results = self.query_index(index_name, query_image, encoder, TryChroma.MAX_RESULTS, constraints)
        logging.info(f"search results:{results}")
        return results

    ##############################################
    # Manage db
    ##############################################
    def build_index(self, index_name: str, embeddings: List[Any], metadata: List[Dict[str, Any]]) -> None:
        logging.info(f"build_index {index_name} {len(embeddings)=} {len(metadata)=}")
        ids = self._ids_from_metadata(metadata)

        self.delete_index(index_name)
        collection = self._chroma_client.create_collection(name=index_name)
        collection.add(
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )
        logging.info(f"Indexed {index_name}: {collection.count()=}")

    def delete_index(self, name: str):
        for idx in self.indexes():
            if idx == name:
                self._chroma_client.delete_collection(name=name)
                logging.info(f"Deleted index {name}")
                break

    def indexes(self) -> List[str]:
        return [c.name for c in self._chroma_client.list_collections()]

    def query_index(self,
                    name: str,
                    query: Image,
                    encoder: str,
                    num_results: int,
                    constraints: Dict[str, str]) -> List[Any]:
        logging.info(f"index into {name} {type(query)=} {encoder=}")
        query_embedding = self.embeddings(encoder, query)
        collection = self._chroma_client.get_collection(name=name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results,
            where=constraints
        )
        return results['metadatas'][0]

    ##############################################
    # Manage Data
    ##############################################

    def load_dataset(self, dataset_name: str) -> Tuple:
        """
        Load HF dataset
        :param dataset_name:
        :return:
        """
        logging.info(f"load_dataset {dataset_name}")
        dataset_path = Path("datasets", dataset_name)
        dataset = datasets.load_from_disk(dataset_path.as_posix())
        images = dataset["image"]
        metadata = dataset.remove_columns("image")
        metadata = metadata.to_pandas().to_dict('records')
        return images, metadata

    def embeddings(self, encoder_name: str, images) -> List[Any]:
        # image resize and keep original?
        encoder = self._collection_encoder[encoder_name]
        return encoder.encode(images).tolist()

    def get_image(self, image_path: str) -> Image:
        im = img.open(image_path)
        im.thumbnail((200,270))
        return im

    def _ids_from_metadata(self, metadata: List[Dict[str, Any]]) -> List[str]:
        return [str(i) for i in pd.DataFrame(metadata)['id'].tolist()]

    def _save_images(self, index_name: str, images: List[object], metadata: List[Dict[str, Any]]) -> None:
        ids = self._ids_from_metadata(metadata)
        dir_path = Path("images", index_name)
        dir_path.mkdir(parents=True, exist_ok=True)
        k: str
        v: Image
        for k,v in zip(ids, images):
            image_path = Path(dir_path, k+"_img.png")
            v.save(image_path)

    @staticmethod
    def _update_encoder_mapping(index_name: str, encoder_name: str):
        mapping = TryChroma._inflate_encoder_mapping()
        mapping[index_name] = encoder_name
        mapping_path = Path("db", "encoder_mapping.json")
        json.dump(mapping, open(mapping_path, 'w'))

    @staticmethod
    def _inflate_encoder_mapping() -> Dict[str, str]:
        mapping_path = Path("db", "encoder_mapping.json")
        return json.load(open("db/encoder_mapping.json")) if mapping_path.exists() else {}

    def parse_results(self, metadata: List[Any], index_name: str) -> List[Image]:
        gallary = []
        dir_path = Path("images", index_name)
        for md in metadata:
            id = str(md['id'])
            image_path = Path(dir_path, id + "_img.png")
            im = img.open(image_path)
            im.thumbnail((200, 270))
            gallary.append(im)
        return gallary

    def display_result(self, metadata: List[Any], index_name: str) -> None:
        for img in self.parse_results(metadata, index_name):
            img.show()


@click.group(
    invoke_without_command=False,
    help="""
    test out chromadb
    """,
    )
@click.pass_context
def cli(ctx):
    pass


@cli.command("ingest")
@click.option(
    "--src",
    type=click.STRING,
    default="scraped/staud.json",
    help="data src for dataset",
)
@click.option(
    "--type",
    type=click.STRING,
    default="json",
    help="[json|hub]",
)
@click.option(
    "--target",
    type=click.STRING,
    default="staud",
    help="target directory within datasets",
)
@click.pass_context
def ingest(ctx, src: str, type: str, target: str):
    """
    Build dataset from file
    """
    TryChroma.ingest(src, type, target)


@cli.command("prepare")
@click.option(
    "--dataset",
    type=click.STRING,
    default="staud",
    help="dataset name (under datasets/)",
)
@click.option(
    "--encoder",
    type=click.STRING,
    default="image_clipb",
    help="Type {image_clipb|image_clipl}",
)
@click.pass_context
def prepare(ctx, dataset: str, encoder: str):
    """
    Build index using given dataset, encoder
    """
    client = TryChroma()
    client.prepare(dataset, encoder)


@cli.command("search")
@click.option(
    "--index",
    type=click.STRING,
    default="staud_image_clipb",
    help="in form: dataset_encoder",
)
@click.option(
    "--image",
    type=click.STRING,
    default=None,
    help="Pathname to query image file (default None)",
)
@click.option(
    "--category",
    type=click.STRING,
    default=None,
    help="Filter category (default None)",
)
@click.option(
    "--show",
    type=click.BOOL,
    default=False,
    help="True to display",
)
@click.pass_context
def search(ctx,
           index: str,
           image: Optional[str] = None,
           category: Optional[str] = None,
           show: Optional[bool] = False):
    """
    Search for images matching given image path
    """
    client = TryChroma()
    query_image = client.get_image(image)
    constraints = {}
    if category is not None:
        constraints = {"subCategory": category}
    results = client.search(query_image, constraints=constraints, index_name=index)
    if show:
        client.display_result(results, index)


if __name__ == "__main__":
    cli()