import os
import requests

from client.base_client import BaseClient


class SolrResp:
    def __init__(self, resp):
        resp = resp.json()
        self.status_code = 400
        if 'responseHeader' in resp and resp['responseHeader']['status'] == 0:
            print("request acknowledged!")
            self.status_code = 200
        else:
            self.status_code = resp['responseHeader']['status']
            self.text = resp['error']['msg']


class SolrClient(BaseClient):
    def __init__(self):
        self.docker = os.environ.get('LTR_DOCKER') != None
        self.solr = requests.Session()

        if self.docker:
            self.host = 'solr'
            self.solr_base_ep = 'http://solr:8983/solr'
        else:
            self.host = 'localhost'
            self.solr_base_ep = 'http://localhost:8983/solr'

    def get_host(self):
        return self.host

    def name(self):
        return "solr"

    @staticmethod
    def resp_msg(msg: str, resp: SolrResp, throw=True):
        print('resp_msg: {} [Status: {}]'.format(msg, resp.status_code))
        if resp.status_code >= 400:
            print(resp.text)
            if throw:
                raise RuntimeError(resp.text)

    def delete_index(self, index):
        params = {
            'action': 'UNLOAD',
            'core': index,
            'deleteIndex': 'true',
            'deleteDataDir': 'true',
            'deleteInstanceDir': 'true'
        }

        resp = requests.get('{}/admin/cores?'.format(self.solr_base_ep), params=params)
        self.resp_msg("Deleted index {}".format(index), SolrResp(resp))

    def create_index(self, index_name, index_spec):
        # Presumes there is a link between the docker container and the 'index'
        # directory under docker/solr/ (ie docker/solr/tmdb/ is linked into
        # Docker container configsets)
        params = {
            'action': 'CREATE',
            'name': index_name,
            'configSet': index_spec,
        }
        resp = requests.get('{}/admin/cores?'.format(self.solr_base_ep), params=params)

        self.resp_msg("Created index {}".format(index_name), SolrResp(resp))

    def index_documents(self, index, doc_src):
        def commit():
            print('Committing changes')
            resp = requests.get('{}/{}/update?commit=true'.format(self.solr_base_ep, index))
            self.resp_msg("Committed index {}".format(index), resp)

        def flush(docs):
            print('Flushing {} docs'.format(len(docs)))
            resp = requests.post('{}/{}/update'.format(
                self.solr_base_ep, index), json=docs)
            self.resp_msg("Done", resp)
            docs.clear()

        BATCH_SIZE = 5000
        docs = []
        for doc in doc_src:
            if 'release_date' in doc and doc['release_date'] is not None:
                doc['release_date'] += 'T00:00:00Z'

            docs.append(doc)

            if len(docs) % BATCH_SIZE == 0:
                flush(docs)

        flush(docs)
        commit()

    def query(self, index, query):
        url = '{}/{}/select?'.format(self.solr_base_ep, index)

        resp = requests.post(url, data=query)
        #resp_msg(msg='Query {}...'.format(str(query)[:20]), resp=resp)
        resp = resp.json()

        qtime = resp['responseHeader']['QTime']
        numfound = resp['response']['numFound']

        return resp['response']['docs'], qtime, numfound

    def analyze(self, index, fieldtype, text):
        # http://localhost:8983/solr/msmarco/analysis/field
        url = '{}/{}/analysis/field?'.format(self.solr_base_ep, index)

        query={
            "analysis.fieldtype": fieldtype,
            "analysis.fieldvalue": text
        }

        resp = requests.post(url, data=query)

        analysis_resp = resp.json()
        tok_stream = analysis_resp['analysis']['field_types']['text_general']['index']
        tok_stream_result = tok_stream[-1]
        return tok_stream_result

    def term_vectors_skip_to(self, index, q='*:*', skip=0):
        url = '{}/{}/tvrh/'.format(self.solr_base_ep, index)
        query={
            'q': q,
            'cursorMark': '*',
            'sort': 'id asc',
            'fl': 'id',
            'rows': str(skip)
            }
        tvrh_resp = requests.post(url, data=query)
        return tvrh_resp.json()['nextCursorMark']

    def get_doc(self, index, doc_id):
        params = {
            'q': 'id:{}'.format(doc_id),
            'wt': 'json'
        }

        resp = requests.post('{}/{}/select'.format(self.solr_base_ep, index), data=params).json()
        return resp['response']['docs'][0]



