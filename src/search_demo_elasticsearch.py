import streamlit as st
from bert_serving.client import BertClient
from client.elastic_client import ElasticClient
from client.utils import get_solr_vector_search, get_elasticsearch_vector
import pandas as pd
import plotly.graph_objects as go

# st.write("Connecting the BertClient...")
from data_utils import compute_sbert_vectors, compute_bert_vectors

bc = BertClient()
# st.write("Connecting the ElasticClient...")
ec = ElasticClient()


# Query config:
# 1. es-vanilla is default dense vector based search, no KNN / ANN involved
# 2. es-elastiknn is elastiknn based KNN search with configurable similarity
def get_query_config(search_method, ranker, distance_metric, query, bc: BertClient, docs_count):
    """
    Compute query config for the given method, ranker function and query
    :param ranker: BERT or SBERT
    :param search_method: one of es-vanilla, es-elastiknn or es-opendistro
    :param distance_metric: only applies to es-vanilla method: cosineSimilarity or dotProduct
    :param query: user query in plain string
    :param bert_client: bert-as-service client to compute query embedding vector
    :return: query config to be executed in Elasticsearch
    """
    es_query = None
    if search_method == 'es-vanilla':
        query_vector = None
        if ranker == "BERT":
            query_vector = get_elasticsearch_vector(compute_bert_vectors(query, bc))
        elif ranker == "SBERT":
            query_vector = get_elasticsearch_vector(compute_sbert_vectors(query))
        es_query = {
            "size": docs_count,
            "_source": ["id", "_text_", "url"],
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": distance_metric,
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }
    elif search_method == 'es-elastiknn':
        query_vector = None
        if ranker == "BERT":
            query_vector = get_elasticsearch_vector(compute_bert_vectors(query, bc))
        elif ranker == "SBERT":
            query_vector = get_elasticsearch_vector(compute_sbert_vectors(query))

        es_query = {
            "size": docs_count,
            "_source": ["id", "_text_", "url"],
            "query": {
                "elastiknn_nearest_neighbors": {
                    "field": "vector",
                    "vec": {
                        "values": query_vector,
                    },
                    "model": "lsh",
                    "similarity": "angular",
                    "candidates": 10
                }
            }
        }
    return es_query


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)


def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)


@st.cache
def _filter_results(results, number_of_rows, number_of_columns) -> pd.DataFrame:
    return results.iloc[0:number_of_rows, 0:number_of_columns]


def plotly_table(results):
    st.header("Plotly Table (go.Table)")
    number_of_rows, number_of_columns, style = 10, 5, True
    filter_table = _filter_results(results, number_of_rows, number_of_columns)

    header_values = list(filter_table.columns)
    cell_values = []
    for index in range(0, len(filter_table.columns)):
        cell_values.append(filter_table.iloc[:, index: index + 1])

    if not style:
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=header_values), cells=dict(values=cell_values)
                )
            ]
        )
    else:
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=header_values, fill_color="paleturquoise", align="left"
                    ),
                    cells=dict(values=cell_values, fill_color="lavender", align="left"),
                )
            ]
        )

    st.plotly_chart(fig)
    st.markdown(
        """
Pros

- Can maximize
- Can transfer and display 10.000 rows and 5 columns in 10000 seconds.
- Can do advanced styling and layout.

Cons

- Cannot sort or filter
- The scrollbar is *thin* and can be difficult to select/ drag.

References:

- [Plotly Table Introduction](https://plot.ly/python/table/)
- [Plotly Table Reference](https://plot.ly/python/reference/#table)
"""
    )


st.title('BERT & Elasticsearch Search Demo')
vector_search_implementation = st.sidebar.radio('Search using method', ['es-vanilla', 'es-elastiknn'], index=0)
index = st.sidebar.selectbox('Target index',
                     ('vector_1000', 'vector_10000', 'vector_100000', 'vector_1000000',
                      'elastiknn_1000', 'elastiknn_10000', 'elastiknn_100000', 'elastiknn_1000000'))
ranker = st.sidebar.radio('Rank by', ["BERT", "SBERT", "BM25"], index=0)
measure = st.sidebar.radio('Ranker distance metric (applies only to BERT/SBERT and es-vanilla)', ["cosine ([0,1])", "dot product (unbounded)"], index=0)

local_css("css/style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

icon("search")

query = st.text_input("Type your query here", "history of humanity")
button_clicked = st.button("Go")
n_docs = st.sidebar.slider(label="Number of Documents to View", min_value=10, max_value=50, value=10, step=10)

if button_clicked and query != "":
    st.write("Ranker: {}".format(ranker))
    st.write("Index: {}".format(index))
    es_query = None
    if ranker == "BERT" or ranker == "SBERT":
        cosine = "false"
        distance_metric = ''
        if measure == "cosine ([0,1])":
            cosine = "true"
            distance_metric = "1.0 + cosineSimilarity(params['query_vector'], 'vector')"
        elif measure == "dot product (unbounded)":
            cosine = "false"
            # Using the standard sigmoid function prevents scores from being negative
            distance_metric = """double value = dotProduct(params.query_vector, 'vector');
          return sigmoid(1, Math.E, -value);"""

        es_query = get_query_config(search_method=vector_search_implementation,
                                    ranker=ranker,
                                    distance_metric=distance_metric,
                                    query=query,
                                    bc=bc,
                                    docs_count=n_docs)
    elif ranker == "BM25":
        es_query = {
            "query": query
        }

    with st.spinner(text="Searching..."):
        docs, query_time, numfound = ec.query(index, es_query)
    st.success("Done!")

    st.write("Query time: {} ms".format(query_time))
    st.write("Found documents: {}".format(numfound))
    if numfound > 0:
        df = pd.DataFrame(docs)
        st.table(df)
        # Try plotly table for different UX, than standard streamlit table rendering
        # plotly_table(df)

        chart_data = pd.DataFrame(
            df["_score"],
            columns=['score'])
        st.line_chart(chart_data)
