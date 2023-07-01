from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import pickle
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
    TYPE_CLIP_COL = "image_clip"
    TYPE_FCLIP_COL = "image_fclip"
    EMBEDDER_CLIP = 'sentence-transformers/clip-ViT-B-32'
    EMBEDDER_FCLIP = 'sentence-transformers/clip-ViT-L-14'
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
        self._embedder_clip = SentenceTransformer(TryChroma.EMBEDDER_CLIP, device=dd)
        self._embedder_fclip = SentenceTransformer(TryChroma.EMBEDDER_FCLIP, device=dd)
        self._collection_embedder = {
            TryChroma.TYPE_CLIP_COL: self._embedder_clip,
            TryChroma.TYPE_FCLIP_COL: self._embedder_fclip
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

    def prepare(self, dataset_name: str, embedder_name: str) -> None:
        """
        Prepare for usage by indexing images from a dataset,
        using embedder based in index name
        :param index_name:
        :return:
        """
        logging.info(f"Prepare loading {dataset_name} embedding {embedder_name}")
        images, metadata = self.load_dataset(dataset_name)
        embeddings = self.embeddings(embedder_name, images)
        self.build_index(dataset_name, embedder_name, embeddings, metadata)
        map_path = Path("db", dataset_name+"_"+embedder_name+".pkl")
        self._persist_map(images, metadata, map_path)

    def search(self,
               query_image: Image,
               constraints: Dict[str, str],
               index_name: Optional[str] = "staud-small_image_clip",
               embedder: Optional[str] = "image_clip") -> List[Dict[str,Any]]:
        """
        Answer back a list of metadata search results from the given Image
        and query constraints
        :param query_image:
        :param constraints:
        :param index_name:
        :return:
        """
        results = self.query_index(index_name, query_image, embedder, TryChroma.MAX_RESULTS, constraints)
        logging.info(f"search results:{results}")
        return results

    ##############################################
    # Manage db
    ##############################################
    def build_index(self, dataset: str, embedder: str, embeddings: List[Any], metadata: List[Dict[str, Any]]) -> None:
        logging.info(f"build_index {dataset}_{embedder} {len(embeddings)=} {len(metadata)=}")
        index_name = f"{dataset}_{embedder}"
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
        for c in self._chroma_client.list_collections():
            if c.name == name:
                self._chroma_client.delete_collection(name=name)
                logging.info(f"Deleted index {name}")
                break

    def query_index(self,
                    name: str,
                    query: Image,
                    embedder: str,
                    num_results: int,
                    constraints: Dict[str, str]) -> List[Any]:
        logging.info(f"index into {name} {type(query)=} {embedder=}")
        query_embedding = self.embeddings(embedder, query)
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
        dataset = datasets.load_from_disk(dataset_path)
        images = dataset["image"]
        metadata = dataset.remove_columns("image")
        metadata = metadata.to_pandas().to_dict('records')
        return images, metadata

    def embeddings(self, embedder_name: str, images) -> List[Any]:
        # image.thumbnail((60, 80), img.Resampling.LANCZOS)
        embedder = self._collection_embedder[embedder_name]
        return embedder.encode(images).tolist()

    def get_image(self, image_path: str) -> Image:
        im = img.open(image_path)
        im.thumbnail((60,80), img.Resampling.LANCZOS)
        return im

    def _ids_from_metadata(self, metadata: List[Dict[str, Any]]) -> List[str]:
        return [str(i) for i in pd.DataFrame(metadata)['id'].tolist()]

    def _persist_map(self, images: List[object], metadata: List[Dict[str, Any]], mapping_path: Path) -> None:
        ids = self._ids_from_metadata(metadata)
        id2image: Dict[str, object] = dict(zip(ids, images))
        with open(mapping_path, 'wb') as f:
            pickle.dump(id2image, f)

    def _inflate_map(self, mapping_path: Path) -> Dict[str,Image]:
        with open(mapping_path, 'rb') as f:
            id2image = pickle.load(f)
        return id2image

    def parse_results(self, metadata: List[Any], index_name: str) -> List[Image]:
        gallary = []
        map_path = Path("db", index_name + ".pkl")
        id2image = self._inflate_map(map_path)
        for md in metadata:
            id = str(md['id'])
            im = id2image[id]
            im.thumbnail((60, 80), img.Resampling.LANCZOS)
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
    default="scraped/staud-small.json",
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
    default="staud-small",
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
    default="staud-small",
    help="dataset name (under datasets/)",
)
@click.option(
    "--embedder",
    type=click.STRING,
    default="image_clip",
    help="Type {image_clip|image_fclip}",
)
@click.pass_context
def prepare(ctx, dataset: str, embedder: str):
    """
    Build collection using given name
    """
    client = TryChroma()
    client.prepare(dataset, embedder)


@cli.command("search")
@click.option(
    "--index",
    type=click.STRING,
    default="staud_image_clip",
    help="{dataset_image_clip|dataset_image_fclip}",
)
@click.option(
    "--embedder",
    type=click.STRING,
    default="image_clip",
    help="{image_clip|image_fclip}",
)
@click.option(
    "--image",
    type=click.STRING,
    default=None,
    help="Pathname to image file (default None)",
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
           embedder: str,
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
    results = client.search(query_image, constraints=constraints, index_name=index, embedder=embedder)
    if show:
        client.display_result(results, index)


if __name__ == "__main__":
    cli()