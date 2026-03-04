from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from fmeh.graph.nodes import (
    NodeContext,
    node_build_prompt,
    node_evaluate,
    node_log,
    node_parse_output,
    node_repair_output,
    node_retrieve_context,
    node_run_model,
    route_after_parse,
)
from fmeh.graph.state import EvalState


def build_eval_graph(ctx: NodeContext):
    graph = StateGraph(EvalState)

    graph.add_node("retrieve_context", node_retrieve_context(ctx))
    graph.add_node("build_prompt", node_build_prompt(ctx))
    graph.add_node("run_model", node_run_model(ctx))
    graph.add_node("parse_output", node_parse_output)
    graph.add_node("repair_output", node_repair_output)
    graph.add_node("evaluate", node_evaluate)
    graph.add_node("log", node_log(ctx))

    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "build_prompt")
    graph.add_edge("build_prompt", "run_model")
    graph.add_edge("run_model", "parse_output")
    graph.add_conditional_edges(
        "parse_output",
        route_after_parse,
        {
            "repair_output": "repair_output",
            "evaluate": "evaluate",
        },
    )
    graph.add_edge("repair_output", "parse_output")
    graph.add_edge("evaluate", "log")
    graph.add_edge("log", END)

    return graph.compile()
