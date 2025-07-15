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
from anomaly_detection_agent import create_anomaly_detection_agent

load_dotenv()  # Load variables from .env

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

model = ChatOpenAI(model="gpt-4o-mini")

influx_host = os.getenv("INFLUX_HOST")
influx_password = os.getenv("INFLUX_PASSWORD")

# Create all agents
data_extraction_agent = create_data_extraction_agent(model)
enhanced_network_agent = create_enhanced_network_agent(model)
anomaly_detection_agent = create_anomaly_detection_agent(model)

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

# Create supervisor with all agents including anomaly detection
workflow = create_supervisor(
    [data_extraction_agent, enhanced_network_agent, xapp_agent, anomaly_detection_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing telecom network analysis with a 4-agent workflow. "
        "Route requests as follows:\n"
        
        "For NETWORK-related queries (UE analysis, cell analysis, mobility, performance):\n"
        "1. FIRST: Always route to data_extraction_agent to filter relevant data\n"
        "2. THEN: Route to enhanced_network_agent to analyze the filtered data\n"
        
        "For ANOMALY DETECTION queries (detect anomalies, find issues, problems, outliers):\n"
        "1. FIRST: Route to data_extraction_agent with the query INCLUDING the word 'anomaly' or 'detect'\n"
        "2. THEN: Route to anomaly_detection_agent with the filtered file path\n"
        "3. The anomaly agent needs ALL data columns to perform proper analysis\n"
        
        "For XAPP-related queries (logs, deployment, configurations):\n"
        "- Route directly to xapp_agent\n"
        
        "IMPORTANT WORKFLOW:\n"
        "- Network queries MUST go through data extraction first, then analysis\n"
        "- Anomaly detection queries MUST go through data extraction first, then anomaly detection\n"
        "- For anomaly detection, ensure the data extraction agent extracts ALL columns, not just anomaly columns\n"
        "- The anomaly_detection_agent will run its own detection algorithms on the full data\n"
        
        "Example routing:\n"
        "Query: 'Detect anomalies for UE5'\n"
        "1. data_extraction_agent → extracts ALL UE5 data (mobility + signal + throughput + resources)\n"
        "2. anomaly_detection_agent → runs comprehensive anomaly detection algorithms\n"
        
        "Query: 'Find signal quality issues in the network'\n"
        "1. data_extraction_agent → extracts all signal quality data with 'detect signal anomaly' query\n"
        "2. anomaly_detection_agent → detects signal anomalies using statistical methods\n"
    )
)

# Compile and run
app = workflow.compile()

def get_data():
    # get_ue_cell_data(influx_host, influx_password)
    # stream_xapps_logs()
    pass

def main():
    print("=== TELECOM NETWORK ANALYSIS SYSTEM ===")
    print("Available capabilities:")
    print("1. Network Analysis: 'Which cells has UE9 travelled through?'")
    print("2. Anomaly Detection: 'Detect anomalies for UE5'")
    print("3. Signal Analysis: 'Find signal quality issues'")
    print("4. Performance Issues: 'Check for throughput problems'")
    print("5. xApp Logs: 'Check ad xApp status'")
    print("\nType 'exit' or 'quit' to end the session.\n")
    
    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting the application.")
            return

        get_data()
        print("Updated data.")
        
        try:
            result = app.invoke({
                "messages": [
                    {
                        "role": "user", 
                        "content": f"{query}."
                    }
                ]
            }, config={"recursion_limit": 50})
            
            print("\n=== ANALYSIS RESULTS ===")
            for r in result['messages']:
                r.pretty_print()
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            print("Please try rephrasing your query or check the data files.")

if __name__ == "__main__":
    main()