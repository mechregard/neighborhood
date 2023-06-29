from typing import List, Dict, Tuple, Optional, Any
import logging
import pickle
import click
import pandas as pd
from PIL import Image as img
from PIL.Image import Image
import torch
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
    ID2IMAGE_MAP_FILENAME = "db/mapping_"
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

    def prepare(self, index_name: str) -> None:
        """
        Prepare for usage by indexing images from a dataset,
        using embedder based in index name
        :param index_name:
        :return:
        """
        logging.info(f"Prepare loading data into {index_name}")
        images, metadata = self.load_dataset()
        embeddings = self.embeddings(index_name, images)
        self.build_index(index_name, embeddings, metadata)
        self._persist_map(images, metadata, index_name)

    def search(self, query_image: Image,
               constraints: Dict[str, str],
               index_name: Optional[str] = "image_clip") -> List[Dict[str,Any]]:
        results = self.query_index(index_name, query_image, TryChroma.MAX_RESULTS, constraints)
        return results

    ##############################################
    # Manage db
    ##############################################
    def build_index(self, name: str, embeddings: List[Any], metadata: List[Dict[str, Any]]) -> None:
        logging.info(f"build_index {name} {len(embeddings)=} {len(metadata)=}")

        ids = self._ids_from_metadata(metadata)

        self.delete_index(name)
        collection = self._chroma_client.create_collection(name=name)
        collection.add(
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )
        logging.info(f"Indexed {name}: {collection.count()=}")

    def delete_index(self, name: str):
        for c in self._chroma_client.list_collections():
            if c.name == name:
                self._chroma_client.delete_collection(name=name)
                logging.info(f"Deleted index {name}")
                break

    def query_index(self, name: str,
                    query: Image,
                    num_results: int,
                    constraints: Dict[str, str]) -> List[Dict[str,Any]]:
        logging.info(f"index into {name} {type(query)=}")
        query_embedding = self.embeddings(name, query)
        collection = self._chroma_client.get_collection(name=name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results,
            where=constraints
        )
        logging.info(f"Query result: {results}")
        return results['metadatas'][0]

    ##############################################
    # Manage Data
    ##############################################
    def load_dataset(self, start: Optional[int] = None, count: Optional[int] = 10) -> Tuple:
        """
        Load HF dataset with optional split and pull out images, metadata
        size is 60,80
        :param start:
        :param count:
        :return:
        """
        split_str = f"train[{start}:{start+count}]" if start is not None else "train"
        logging.info(f"load_dataset using split: {split_str}")
        dataset = load_dataset(
            TryChroma.DATASET_NAME,
            split=split_str
        )
        images = dataset["image"]
        metadata = dataset.remove_columns("image")
        metadata = metadata.to_pandas().to_dict('records')
        return images, metadata

    def embeddings(self, index_name: str, images) -> List[Any]:
        embedder = self._collection_embedder[index_name]
        return embedder.encode(images).tolist()

    def get_image(self, image_path: Optional[str] = None) -> Image:
        if image_path is None:
            images, _  = self.load_dataset(start=800, count=1)
            return images[0]
        else:
            im = img.Image.open(image_path)
            im.thumbnail((60,80), Image.Resampling.LANCZOS)
            return im

    def _ids_from_metadata(self, metadata: List[Dict[str, Any]]) -> List[str]:
        return [str(i) for i in pd.DataFrame(metadata)['id'].tolist()]

    def _persist_map(self, images: List[object], metadata: List[Dict[str, Any]], index_name: str) -> None:
        ids = self._ids_from_metadata(metadata)
        id2image: Dict[str, Image] = dict(zip(ids, images))
        with open(TryChroma.ID2IMAGE_MAP_FILENAME+index_name+".pkl", 'wb') as f:
            pickle.dump(id2image, f)

    def _inflate_map(self, index_name: str) -> Dict[str,Image]:
        with open(TryChroma.ID2IMAGE_MAP_FILENAME+index_name+".pkl", 'rb') as f:
            id2image = pickle.load(f)
        return id2image

    def parse_results(self, metadata: List[Any], index_name: str) -> List[Image]:
        gallary = []
        id2image: Dict[str, Image] = self._inflate_map(index_name)
        for md in metadata:
            id = str(md['id'])
            gallary.append(id2image[id])
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


@cli.command("prepare")
@click.option(
    "--index",
    type=click.STRING,
    default="image_clip",
    help="Type {image_clip|image_fclip}",
)
@click.pass_context
def prepare(ctx, index: str):
    """
    Build collection using given name
    """
    client = TryChroma()
    client.prepare(index)


@cli.command("search")
@click.option(
    "--index",
    type=click.STRING,
    default="image_clip",
    help="Type {image_clip|image_fclip}",
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
@click.pass_context
def search(ctx, index: str, image: Optional[str] = None, category: Optional[str] = None):
    """
    Search for images matching given image path
    """
    client = TryChroma()
    query_image = client.get_image(image)
    constraints = {}
    if category is not None:
        constraints = {"subCategory": category}
    results = client.search(query_image, constraints=constraints, index_name=index)
    client.display_result(results, index)


if __name__ == "__main__":
    cli()