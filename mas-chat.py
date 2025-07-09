from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
import pandas as pd
from dotenv import load_dotenv
from utils import get_ue_cell_data, stream_xapps_logs
import os
from data_extraction_agent import (
    create_data_extraction_agent,
    create_enhanced_network_agent,
    read_filtered_data
)

load_dotenv()  # Load variables from .env

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

model = ChatOpenAI(model="gpt-4o-mini")


influx_host = os.getenv("INFLUX_HOST")
influx_password = os.getenv("INFLUX_PASSWORD")



# def get_ue_data(file_path: str) -> dict: 
#     """
#     Reads the content of a UEs csv file.
#     """
#     try:
#         df = pd.read_csv(file_path)
#         return df.to_dict(orient='records')
#     except Exception as e:
#         return f"Error reading file {file_path}: {str(e)}"
    
# def get_cell_data(file_path: str) -> dict:
#     """
#     Reads the content of a cells csv file.
#     """
#     try:
#         df = pd.read_csv(file_path)
#         return df.to_dict(orient='records')
#     except Exception as e:
#         return f"Error reading file {file_path}: {str(e)}"

data_extraction_agent = create_data_extraction_agent(model)
enchanced_network_agent = create_enhanced_network_agent(model)

def read_log_file(file_path: str) -> str:
    """
    Reads the content of a log file.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

xapp_agent = create_react_agent(
    model=model,
    name="xapp_agent",
    prompt=(
        "You are an expert in analyzing xApp logs for qp, ad, and trafficxapp applications. "
        "When asked about xApp status, immediately read the relevant log files: "
        "- For ad xApp: read_log_file('data/logs/ad.log') "
        "- For qp xApp: read_log_file('data/logs/qp.log') "
        "- For trafficxapp: read_log_file('data/logs/trafficxapp.log') "
        "Analyze the logs for errors, warnings, and overall health status. "
        "Provide specific findings about what you discover in the logs."
    ),
    tools=[read_log_file],
)

workflow = create_supervisor(
    [data_extraction_agent, enchanced_network_agent, xapp_agent],
    model=model,
    prompt=(
    "You are a team supervisor managing telecom network analysis with a 3-agent workflow. "
    "Route requests as follows:\n"
    
    "For NETWORK-related queries (UE analysis, cell analysis, mobility, performance):\n"
    "1. FIRST: Always route to data_extraction_agent to filter relevant data\n"
    "2. THEN: Route to enhanced_network_agent to analyze the filtered data\n"
    
    "For XAPP-related queries (logs, deployment, configurations):\n"
    "- Route directly to xapp_agent\n"
    
    "IMPORTANT WORKFLOW:\n"
    "- Network queries MUST go through data extraction first, then analysis\n"
    "- Never send network queries directly to enhanced_network_agent without filtered data\n"
    "- The data_extraction_agent will create filtered CSV files\n"
    "- The enhanced_network_agent will analyze these filtered files\n"
    
    "Example routing:\n"
    "Query: 'Which cell towers has UE10 passed through?'\n"
    "1. data_extraction_agent → extracts UE10 mobility data\n"
    "2. enhanced_network_agent → analyzes the filtered data\n"
)
)

# Compile and run
app = workflow.compile()

def get_data():
    # get_ue_cell_data(influx_host, influx_password)
    # stream_xapps_logs()
    pass

def main():
    # get the query as input from terminal
    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting the application.")
            return

        get_data()
        print("Updated data.")
        result = app.invoke({
            "messages": [
                {
                    "role": "user", 
                    "content": f"{query}."
                }
            ]
        }, config={"recursion_limit":24})
        print("\n=== NETWORK PERFORMANCE CHECK ===")
        for r in result['messages']:
            r.pretty_print()


if __name__ == "__main__":
    main()