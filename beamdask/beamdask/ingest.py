import apache_beam as beam
from typing import List, Dict, Tuple, Optional, Any
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.dataframe.io import read_json
from apache_beam.dataframe.convert import to_pcollection
import pyarrow
from pathlib import Path


class ImageResize(beam.DoFn):
    def __init__(self, catalog: str, image_size=(224, 224)):
        self.catalog = catalog
        self.image_size = image_size

    def process(self, element):
        import pathlib
        from PIL import Image
        import requests
        url = element.image
        save_path = pathlib.Path("images", self.catalog, element.id + ".jpg")
        try:
            image_src = Image.open(requests.get(url, stream=True).raw)
            image_src.thumbnail(self.image_size, Image.Resampling.LANCZOS)
            image_src.save(save_path.as_posix(), "JPEG")
        except Exception as e:
            print(f"Failed processing {element}: {e}")
        return [element]


def schema_to_dictionary(entry, category: str) -> Dict:
    return {
        'id': entry.id,
        'category': category,
        'gender': entry.gender,
        'masterCategory': entry.masterCategory,
        'subCategory': entry.subCategory,
        'baseColour': entry.baseColour,
        'productDisplayName': entry.productDisplayName,
        'image': entry.image,
        'processed_image': f"images/{category}/{entry.id}.jpg"
    }


def ingest(pipeline, category: str):
    """
    """
    input_path = Path("scrape", category+".json").as_posix()
    output_path = Path("parquet", category).as_posix()
    output_schema = pyarrow.schema(
                                     [
                                         ('id', pyarrow.string()),
                                         ('gender', pyarrow.string()),
                                         ('masterCategory', pyarrow.string()),
                                         ('subCategory', pyarrow.string()),
                                         ('baseColour', pyarrow.string()),
                                         ('productDisplayName',
                                          pyarrow.string()),
                                         ('image', pyarrow.string()),
                                         ('processed_image', pyarrow.string())
                                     ]
                                 )

    with pipeline as p:
        df = p | "Read data" >> read_json(input_path)
        df_collection = to_pcollection(df)
        df_collection | 'Resize' >> beam.ParDo(ImageResize(category)) \
        | "Tuple to Dict for Parquet" >> beam.Map(schema_to_dictionary, category) \
        | 'Write' >> beam.io.parquetio.WriteToParquet(output_path, output_schema, file_name_suffix=".parquet")


if __name__ == "__main__":
    category  = "staud-small"
    Path("images", category).mkdir(parents=True, exist_ok=True)
    beam_options = PipelineOptions(
        runner="DirectRunner",
        direct_num_workers=0,
        direct_running_mode="multi_processing",
    )
    pipeline = beam.Pipeline(options=beam_options)
    ingest(pipeline, category)
