import streamlit as st
from bert_serving.client import BertClient
from client.solr_client import SolrClient
from util.utils import get_solr_vector_search
import pandas as pd
import plotly.graph_objects as go

st.write("Connecting the BertClient...")
bc = BertClient()
st.write("Connecting the SolrClient...")
sc = SolrClient()


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
        cell_values.append(filter_table.iloc[:, index : index + 1])

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


st.title('BERT & Solr Search Demo')
ranker = st.sidebar.radio('Rank by', ["BERT", "BM25"], index=0)
measure = st.sidebar.radio('BERT ranker formula', ["cosine ([0,1])", "dot product (unbounded)"], index=0)

local_css("css/style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

icon("search")

query = st.text_input("Type your query here", "history of humanity")
button_clicked = st.button("Go")
n = st.sidebar.slider(label="Number of Documents to View", min_value=10, max_value=50, value=10, step=10)

if button_clicked or query != "":
    st.write("Query: {}".format(query))
    st.write("Ranker: {}".format(ranker))
    if ranker == "BERT":
        cosine = "false"
        if measure == "cosine ([0,1])":
            cosine = "true"
        elif measure == "dot product (unbounded)":
            cosine = "false"
        query = {
            "q": '{!vp f=vector vector="' + get_solr_vector_search(bc, query) + '" cosine=' + cosine + '}',
            "wt": "json",
            "fl": "id,_text_,url,score",
            "rows": n

        }
    elif ranker == "BM25":
        query = {
            "q": query,
            "wt": "json",
            "fl": "id,_text_,url,score",
            "rows": n

        }
    with st.spinner(text="Searching..."):
        docs, query_time, numfound = sc.query("vector", query)
    st.success("Done!")

    st.write("Query time: {} ms".format(query_time))
    st.write("Found documents: {}".format(numfound))
    df = pd.DataFrame(docs)
    st.table(df)
    # Try plotly table for different UX, than standard streamlit table rendering
    # plotly_table(df)

    chart_data = pd.DataFrame(
        df["score"],
        columns=['score'])
    st.line_chart(chart_data)

