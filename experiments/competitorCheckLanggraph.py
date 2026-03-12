import os
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod # Corrected import location
from tavily import TavilyClient
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define state structure
class CarwashState(TypedDict):
    name: str
    address: str
    features_to_check: List[str]
    search_results: Dict[str, List[dict]]
    feature_present: Dict[str, bool]

# Initialize state
def create_initial_state():
    return {
        "name": "Mister Car Wash",
        "address": "8100 San Pedro Dr NE, Albuquerque, NM 87113, USA",
        "features_to_check": ["Sliding Tunnel", "Vacuum Station"],
        "search_results": {},
        "feature_present": {}
    }

# Initialize Azure LLM
llm = AzureChatOpenAI(
    model="gpt-4o",
    api_version="2024-08-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
    # temperature=0,
)

# Node: Search for feature (unchanged)
def search_feature(state: CarwashState, feature: str):
    query = f"{state['name']} {state['address']} {feature}"
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(query)
    print(f"Query: {query} \n Results: {results}")
    state["search_results"][feature] = results["results"]
    return state

# Node: Search for Sliding Tunnel
def search_sliding_tunnel(state: CarwashState):
    print("Executing Node: search_sliding_tunnel")
    return search_feature(state, "Sliding Tunnel")

# Node: Search for Vacuum Station
def search_vacuum_station(state: CarwashState):
    print("Executing Node: search_vacuum_station")
    return search_feature(state, "Vacuum Station")

# LLM Validation Helper
def validate_with_llm(content: str, feature: str) -> bool:
    if not content.strip():
        return False
        
    prompt = f"""
    Based on the following text, does this Carwash have a '{feature}'?
    Answer ONLY with 'yes' or 'no' (lowercase).

    TEXT: {content}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip().lower() == "yes"
    except Exception as e:
        print(f"LLM Error: {e}")
        return False

# Node: Validate Sliding Tunnel with LLM
def check_sliding_tunnel(state: CarwashState):
    print("Executing Node: check_sliding_tunnel")
    combined_content = " ".join(
        result.get("content", "") for result in state["search_results"].get("Sliding Tunnel", [])
    )
    state["feature_present"]["Sliding Tunnel"] = validate_with_llm(combined_content, "Sliding Tunnel")
    return state

# Node: Validate Vacuum Station with LLM
def check_vacuum_station(state: CarwashState):
    print("Executing Node: check_vacuum_station")
    combined_content = " ".join(
        result.get("content", "") for result in state["search_results"].get("Vacuum Station", [])
    )
    state["feature_present"]["Vacuum Station"] = validate_with_llm(combined_content, "Vacuum Station")
    return state

# Output node (unchanged)
def output_results(state: CarwashState):
    print("Executing Node: output_results")
    print("🔍 Results for Carwash:\n")
    for feature in state["features_to_check"]:
        present = state["feature_present"].get(feature, False)
        status = "✅ Yes" if present else "❌ No"
        print(f"Feature: {feature}")
        print(f"Available: {status}")
        print("Supporting Data:")
        for result in state["search_results"].get(feature, []):
            print(f"  - {result.get('content', '')} [Source: {result.get('url', '')}]")
        print("\n---\n")
    return state

# Build LangGraph workflow
workflow = StateGraph(CarwashState)

# Add nodes
workflow.add_node("search_sliding_tunnel", search_sliding_tunnel)
workflow.add_node("check_sliding_tunnel", check_sliding_tunnel)
workflow.add_node("search_vacuum_station", search_vacuum_station)
workflow.add_node("check_vacuum_station", check_vacuum_station)
workflow.add_node("output_results", output_results)

# Define edges
workflow.add_edge("__start__", "search_sliding_tunnel")
workflow.add_edge("search_sliding_tunnel", "check_sliding_tunnel")
workflow.add_edge("check_sliding_tunnel", "search_vacuum_station")
workflow.add_edge("search_vacuum_station", "check_vacuum_station")
workflow.add_edge("check_vacuum_station", "output_results")

# Compile and run
app = workflow.compile()

# Save graph visualization locally using Pyppeteer
# print("Generating graph visualization...")
# try:
#     with open("graph.png", "wb") as f:
#         f.write(app.get_graph(xray=True).draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER))
#     print("Graph saved to graph.png")
# except ImportError:
#     print("Pyppeteer not installed. Skipping graph generation.")
#     print("Install it with: pip install pyppeteer")
# except Exception as e:
#     print(f"Error generating graph: {e}")

# Execute
print("\nStarting workflow execution...")
initial_state = create_initial_state()
final_state = app.invoke(initial_state)
