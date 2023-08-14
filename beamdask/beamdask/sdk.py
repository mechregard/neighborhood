from pathlib import Path
import logging
from PyPDF2 import PdfReader
import spacy
from typing import List,Dict,Any
import apache_beam as beam
import pyarrow


class ExtractTextFn(beam.DoFn):
    def process(self, element):
        return [self.extract(element)]

    def extract(self, src: str):
        logging.info(f"{src=} ")
        src_path = Path(src)
        with open(src_path, 'rb') as f:
            reader = PdfReader(f)
            text = reader.pages[0].extract_text()
        return text


class SplitTextFn(beam.DoFn):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def process(self, element):
        return self.split(element)

    def split(self, text: str) -> List[Dict[str,Any]]:
        doc = self.nlp(text)
        d = [{"doc": "docid", "text":sent.text} for sent in doc.sents]
        for di in d:
            print(di)
        return d


class Pipeline:
    def __init__(self):
        """
        """
        logging.basicConfig(format='%(levelname)s %(asctime)s %(module)s: %(message)s',
                          datefmt='%Y-%m-%d,%H:%M:%S',
                          level=logging.INFO)
        logging.info(f"Init")
        self.nlp = spacy.load('en_core_web_sm')
        self.pipeline = beam.Pipeline()

    def define(self, files: List[str] ):
        """
         | 'Write to File' >> beam.io.WriteToText("output", append_trailing_newlines=False)
        """
        text = self.pipeline | 'Read PDFs' >> beam.Create(files) \
               | 'Extract Text' >> beam.ParDo(ExtractTextFn()) \
               | 'Split Text' >> beam.ParDo(SplitTextFn()) \
               | 'Write' >> beam.io.WriteToParquet("output",
                                      pyarrow.schema(
                                          [('doc', pyarrow.binary()), ('text', pyarrow.binary())]
                                        ),
                                      file_name_suffix = "parquet"
                                      )

    def run(self):
        result = self.pipeline.run()
        result.wait_until_finish()


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.define(["test.pdf"])
    pipeline.run()
