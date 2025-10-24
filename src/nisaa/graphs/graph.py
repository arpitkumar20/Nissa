from langgraph.graph import END, START, StateGraph
from src.nisaa.graphs.state import ChatBotState
from src.nisaa.graphs.node import  embed_query , top_k_finding, prompt_builder_node, invoke_model_node

def  chat_bot_graph():
    print(" Into Graph ")
    workflow = StateGraph(ChatBotState)
    workflow.add_node("embed_query_node", embed_query)
    workflow.add_node("top_k_node", top_k_finding)
    workflow.add_node("prompt_builder_node", prompt_builder_node)
    workflow.add_node("invoke_model_node", invoke_model_node)

    workflow.add_edge(START, "embed_query_node")
    workflow.add_edge("embed_query_node", "top_k_node")
    workflow.add_edge("top_k_node", "prompt_builder_node")
    workflow.add_edge("prompt_builder_node", "invoke_model_node")
    workflow.add_edge("invoke_model_node", END)

    return workflow.compile()