# -*- coding: utf-8 -*-

"""Get triples from the UMLS dataset."""

import pathlib

from docdata import parse_docdata

from ..base import PathDataset
from ...triples import CoreTriplesFactory, TriplesFactory

__all__ = [
    "UMLS_TRAIN_PATH",
    "UMLS_TEST_PATH",
    "UMLS_VALIDATE_PATH",
    "UMLS",
]

HERE = pathlib.Path(__file__).resolve().parent


@parse_docdata
class UMLS(PathDataset):
    """The UMLS dataset.

    ---
    name: Unified Medical Language System
    statistics:
        entities: 135
        relations: 46
        training: 5216
        testing: 661
        validation: 652
        triples: 6529
    citation:
        author: Zhenfeng Lei
        year: 2017
        github: ZhenfengLei/KGDatasets
    """

    def __init__(self, create_inverse_triples: bool = False, version="default", **kwargs):
        """Initialize the UMLS dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PathDataset`.
        """
        self.version = version
        if version == "default":
            UMLS_TRAIN_PATH = HERE.joinpath("train.txt")
            UMLS_TEST_PATH = HERE.joinpath("test.txt")
            UMLS_VALIDATE_PATH = HERE.joinpath("valid.txt")
        elif version == "rnnlogic":
            UMLS_PATH = '/data/chenxingran/CIBLE/CIBLE-v2/data/umls'
            UMLS_TRAIN_PATH = f"{UMLS_PATH}/train.txt"
            UMLS_TEST_PATH = f"{UMLS_PATH}/test.txt"
            UMLS_VALIDATE_PATH = f"{UMLS_PATH}/valid.txt"
            
            self.temp_entity_to_id = dict()
            with open(f"{UMLS_PATH}/entities.dict", "r") as f:
                for line in f.readlines():
                    id, entity = line.strip().split("\t")
                    self.temp_entity_to_id[entity] = int(id)

            self.temp_relation_to_id = dict()
            with open(f"{UMLS_PATH}/relations.dict", "r") as f:
                for line in f.readlines():
                    id, relation = line.strip().split("\t")
                    self.temp_relation_to_id[relation] = int(id)          


        elif version == "neural-lp":
            UMLS_TRAIN_PATH = "data/umls-neural-lp/train.txt"
            UMLS_TEST_PATH = "data/umls-neural-lp/test.txt"
            UMLS_VALIDATE_PATH = "data/umls-neural-lp/valid.txt"

            self.temp_entity_to_id = dict()
            with open("data/umls-neural-lp/entities.txt", "r") as f:
                for id, line in enumerate(f.readlines()):
                    entity = line.strip()
                    self.temp_entity_to_id[entity] = int(id)

            self.temp_relation_to_id = dict()
            with open("data/umls-neural-lp/relations.txt", "r") as f:
                for id, line in enumerate(f.readlines()):
                    relation = line.strip()
                    self.temp_relation_to_id[relation] = int(id)          

        else:
            raise NotImplementedError
        super().__init__(
            training_path=UMLS_TRAIN_PATH,
            testing_path=UMLS_TEST_PATH,
            validation_path=UMLS_VALIDATE_PATH,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )

    def _load(self) -> None:
        self._training = TriplesFactory.from_path(
            path=self.training_path,
            create_inverse_triples=self.create_inverse_triples,
            load_triples_kwargs=self.load_triples_kwargs,
            entity_to_id=self.temp_entity_to_id if self.version != "default" else None,
            relation_to_id=self.temp_relation_to_id if self.version != "default" else None,
        )

        self._testing = TriplesFactory.from_path(
            path=self.testing_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
            # do not explicitly create inverse triples for testing; this is handled by the evaluation code
            create_inverse_triples=False,
            load_triples_kwargs=self.load_triples_kwargs,
        )


if __name__ == "__main__":
    UMLS().summarize()
