from tavily import TavilyClient
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def search(query: str):
    query = query
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    # results = client.search(query, search_depth='advanced', max_results=3)
    results = client.search(query)

    print(f"\n🔍 Search Query: {query}\n")
    
    for i, result in enumerate(results['results'], 1):
        print(f"{'='*60}")
        print(f"Result {i}")
        print(f"{'-'*60}")
        print(f"📌 Title: {result['title']}")
        print(f"🔗 URL: {result['url']}")
        print(f"📊 Relevance Score: {result['score']:.4f}")
        print(f"📝 Snippet: {result['content']}")
        print(f"{'='*60}\n")

search('Does Champion Xpress Carwash offer services like vacuum controller and sliding tunnel(exterior) - express exterior?')