from abc import ABC, abstractmethod

'''
    This project demonstrates working with LTR in Elasticsearch and Solr

    The goal of this class is to abstract away the server and highlight the steps
    required to begin working with LTR.  This keeps the examples agnostic about
    which backend is being used, but the implementations of each client
    should be useful references to those getting started with LTR on
    their specific platform
'''


class BaseClient(ABC):
    @abstractmethod
    def get_host(self):
        pass

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def delete_index(self, index):
        pass

    @abstractmethod
    def create_index(self, index):
        pass

    @abstractmethod
    def index_documents(self, index, doc_src):
        pass

    @abstractmethod
    def query(self, index, query):
        pass

    @abstractmethod
    def get_doc(self, doc_id):
        pass

