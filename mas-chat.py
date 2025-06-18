from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
import pandas as pd
from dotenv import load_dotenv
from utils import get_ue_cell_data, stream_xapps_logs
import os

load_dotenv()  # Load variables from .env

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

model = ChatOpenAI(model="gpt-4o-mini")


influx_host = os.getenv("INFLUX_HOST")
influx_password = os.getenv("INFLUX_PASSWORD")



def get_ue_data(file_path: str) -> dict: 
    """
    Reads the content of a UEs csv file.
    """
    try:
        df = pd.read_csv(file_path)
        return df.to_dict(orient='records')
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"
    
def get_cell_data(file_path: str) -> dict:
    """
    Reads the content of a cells csv file.
    """
    try:
        df = pd.read_csv(file_path)
        return df.to_dict(orient='records')
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

network_agent = create_react_agent(
    model=model,
    name="network_agent",
    prompt=(
        "You are an expert in telecom network performance. "
        "You can answer questions about UEs and cells data. "
        "Use get_ue_data('data/kpis/ue.csv') to read UE data and get_cell_data('data/kpis/cell.csv') to read cell data. "
        "Always analyze the data and provide specific insights about network performance."
    ),
    tools=[get_ue_data, get_cell_data],
)

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
    [network_agent, xapp_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing xApp and telecom network experts. "
        "Route requests as follows: "
        "- For xApp status, logs, deployment issues, or configurations: use xapp_agent "
        "- For network performance, UE analysis, or cell analysis: use network_agent "
        "Be specific in your routing and ensure agents take action immediately."
    )
)

# Compile and run
app = workflow.compile()

def get_data():
    get_ue_cell_data(influx_host, influx_password)
    stream_xapps_logs()
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
        })
        print("\n=== NETWORK PERFORMANCE CHECK ===")
        for r in result['messages']:
            r.pretty_print()


if __name__ == "__main__":
    main()