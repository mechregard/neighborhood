import apache_beam as beam
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.dataframe.io import read_json
from apache_beam.dataframe.convert import to_pcollection
import pyarrow


class ImageLoad(beam.DoFn):
    def process(self, element):
        from PIL import Image
        path = element["processed_image"]
        image_src = Image.open(path)
        return [image_src]


class ImageEncode(beam.DoFn):
    def process(self, element):
        print(element)
        return [element]


class TextLoad(beam.DoFn):
    def process(self, element):
        desc = f"{element['productDisplayName']}. This is a {element['baseColour']} {element['subCategory']} in {element['masterCategory']} for {element['gender']}"
        return [desc]


class TextEncode(beam.DoFn):
    def process(self, element):
        print(element)
        return [element]


def encode(pipeline, category):
    """
    read parquet to collection
      get image -> encode -> insert
      get text -> encode -> insert
    """
    input_path = Path("parquet", category+"-00000-of-00001.parquet").as_posix()
    with pipeline as p:
        data = p | "Read data" >> beam.io.parquetio.ReadFromParquet(input_path)
        data | 'Image' >> beam.ParDo(ImageLoad()) \
             | 'ImageEncode' >> beam.ParDo(ImageEncode())
        data | 'Text' >> beam.ParDo(TextLoad()) \
             | 'TextEncode' >> beam.ParDo(TextEncode())


if __name__ == "__main__":
    category  = "staud-small"
    Path("images", category).mkdir(parents=True, exist_ok=True)
    beam_options = PipelineOptions(
        runner="DirectRunner",
        direct_num_workers=0,
        direct_running_mode="multi_processing",
    )
    pipeline = beam.Pipeline(options=beam_options)
    encode(pipeline, category)
