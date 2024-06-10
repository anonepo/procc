from transformers import AutoTokenizer
from elasticsearch import Elasticsearch, helpers, exceptions
import json
import time
from loguru import logger

class ESBM25:
    def __init__(self, code_snippets, index_name):
        self.es_host = "http://localhost:9200"
        self.es = Elasticsearch([self.es_host])

        self.index_name = index_name

        logger.info(f"ESBM25 index name: {self.index_name}")

        index_settings = {
            "settings": {"number_of_shards": 3, "number_of_replicas": 1},
            "mappings": {
                "properties": {
                    "code": {"type": "text"},
                    "language": {"type": "keyword"},
                    "path": {"type": "keyword"},
                    "start_line": {"type": "text"},
                }
            },
        }

        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

        if self.es.indices.exists(index=self.index_name):
            logger.info(f"Using exist bm25 index: {self.index_name}")
            doc_count = self._document_count()
            assert doc_count == len(
                code_snippets
            ), "Code Snippets is not equal with the exist, please check!"
            logger.info(f"BM25 Index {self.index_name} have {doc_count} documents")

        else:
            logger.info(f"Create bm25 elasticsearch index {self.index_name}")
            self.es.indices.create(
                index=self.index_name, body=index_settings, ignore=400
            )
            self.index_documents(code_snippets)

    def _document_count(self):
        doc_count = self.es.count(index=self.index_name)["count"]
        return doc_count

    def index_documents(self, code_snippets):
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "code": snippet["code"],
                    "language": snippet["language"],
                    "path": snippet["path"],
                    "start_line": snippet["start_line"],
                },
            }
            for snippet in code_snippets
        ]
        try:
            helpers.bulk(self.es, actions)
        except helpers.BulkIndexError as e:
            pass


    def query_top_k(self, top_k, code):
        query_body = {
            "query": {"match": {"code": code}},
            "size": top_k,
        }

        try:
            response = self.es.search(index=self.index_name, body=query_body)
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append(
                    {
                        "code": source.get("code", ""),
                        "language": source.get("language", ""),
                        "path": source.get("path", ""),
                        "start_line": int(source.get("start_line", 0)),
                        "distance": hit["_score"],
                    }
                )
            return results

        except exceptions.BadRequestError as e:
            print(e)
            return []
