

import time
from pyvis.network import Network
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer


llm = ChatOllama(model="mistral:latest", verbose=True)

# llm_transformer = LLMGraphTransformer(llm=llm)
llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Country", "Organization"],
    allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
    # ignore_tool_usage=True
)


text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
"""
documents = [Document(page_content=text)]
start = time.time()
graph_documents = llm_transformer_filtered.convert_to_graph_documents(
    documents)

print(f"Time taken:{time.time()-start}")
print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")


def draw_graph(graph_documents):
    G = Network()
    for node in graph_documents[0].nodes:
        G.add_node(node.id, label=node.id,
                   properties=node.properties, size=20, title=node.id)
    for relationship in graph_documents[0].relationships:
        G.add_edge(relationship.source.id, relationship.target.id, width=2, color="red",
                   label=relationship.type, properties=relationship.properties)
    return G


g = draw_graph(graph_documents=graph_documents)

g.show("graph.html", notebook=False)
