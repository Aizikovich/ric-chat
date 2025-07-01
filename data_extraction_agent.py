import pandas as pd
import os
from typing import Dict, List
from langgraph.prebuilt import create_react_agent

def extract_ue_mobility_data(query_description: str, ue_id: str = None) -> str:
    """
    Extracts relevant UE mobility data based on query description.
    Returns path to filtered CSV file.
    
    Args:
        query_description: Description of what data is needed
        ue_id: Specific UE ID to filter for (optional)
    """
    try:
        # Read the full UE dataset
        df = pd.read_csv('data/kpis/ue.csv')
        
        # Define column mappings for different query types
        mobility_columns = ['time', 'ue-id', 'nrCellIdentity', 'x', 'y', 'step']
        signal_quality_columns = ['time', 'ue-id', 'nrCellIdentity', 'RF.serving.RSRP', 'RF.serving.RSRQ', 'RF.serving.RSSINR']
        neighbor_cell_columns = ['time', 'ue-id', 'nrCellIdentity'] + [f'nbCellIdentity_{i}' for i in range(4)] + [f'rsrp_nb{i}' for i in range(5)]
        throughput_columns = ['time', 'ue-id', 'nrCellIdentity', 'DRB.UEThpDl', 'targetTput', 'RRU.PrbUsedDl']
        anomaly_columns = ['time', 'ue-id', 'nrCellIdentity', 'Viavi.UE.anomalies', 'x', 'y']
        
        # Determine relevant columns based on query description
        query_lower = query_description.lower()
        
        if any(keyword in query_lower for keyword in ['cell tower', 'passed through', 'mobility', 'movement', 'handover']):
            selected_columns = mobility_columns
            output_suffix = 'mobility'
        elif any(keyword in query_lower for keyword in ['signal', 'rsrp', 'rsrq', 'rssinr', 'quality']):
            selected_columns = signal_quality_columns
            output_suffix = 'signal'
        elif any(keyword in query_lower for keyword in ['neighbor', 'neighbouring', 'adjacent']):
            selected_columns = neighbor_cell_columns
            output_suffix = 'neighbors'
        elif any(keyword in query_lower for keyword in ['throughput', 'performance', 'speed', 'prb']):
            selected_columns = throughput_columns
            output_suffix = 'throughput'
        elif any(keyword in query_lower for keyword in ['anomaly', 'anomalies', 'error', 'problem']):
            selected_columns = anomaly_columns
            output_suffix = 'anomalies'
        else:
            # Default to mobility data for general queries
            selected_columns = mobility_columns
            output_suffix = 'general'
        
        # Filter columns that actually exist in the dataframe
        available_columns = [col for col in selected_columns if col in df.columns]
        
        # Filter the dataframe
        filtered_df = df[available_columns].copy()
        
        # Filter by specific UE if provided
        if ue_id:
            filtered_df = filtered_df[filtered_df['ue-id'] == ue_id]
            output_suffix += f'_{ue_id}'
        
        # Sort by time for chronological analysis
        if 'time' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('time')
        
        # Create output directory if it doesn't exist
        os.makedirs('data/filtered', exist_ok=True)
        
        # Save filtered data
        output_path = f'data/filtered/ue_{output_suffix}.csv'
        filtered_df.to_csv(output_path, index=False)
        
        return f"Filtered data saved to {output_path}. Shape: {filtered_df.shape}. Columns: {list(filtered_df.columns)}"
        
    except Exception as e:
        return f"Error extracting UE data: {str(e)}"

def extract_cell_performance_data(query_description: str, cell_id: int = None) -> str:
    """
    Extracts relevant cell performance data based on query description.
    Returns path to filtered CSV file.
    
    Args:
        query_description: Description of what data is needed
        cell_id: Specific cell ID to filter for (optional)
    """
    try:
        # Read the full cell dataset
        df = pd.read_csv('data/kpis/cell.csv')
        
        # Define column mappings for different query types
        resource_usage_columns = ['time', 'nrCellIdentity', 'availPrbDl', 'availPrbUl', 'measPeriodPrb']
        throughput_columns = ['time', 'nrCellIdentity', 'throughput', 'pdcpBytesDl', 'pdcpBytesUl', 'measPeriodPdcpBytes']
        location_columns = ['time', 'nrCellIdentity', 'x', 'y']
        all_performance_columns = ['time', 'nrCellIdentity', 'throughput', 'availPrbDl', 'availPrbUl', 'pdcpBytesDl', 'pdcpBytesUl']
        
        # Determine relevant columns based on query description
        query_lower = query_description.lower()
        
        if any(keyword in query_lower for keyword in ['resource', 'prb', 'utilization', 'usage']):
            selected_columns = resource_usage_columns
            output_suffix = 'resources'
        elif any(keyword in query_lower for keyword in ['throughput', 'performance', 'bytes', 'data']):
            selected_columns = throughput_columns
            output_suffix = 'throughput'
        elif any(keyword in query_lower for keyword in ['location', 'position', 'coordinates']):
            selected_columns = location_columns
            output_suffix = 'location'
        else:
            # Default to all performance metrics
            selected_columns = all_performance_columns
            output_suffix = 'performance'
        
        # Filter columns that actually exist in the dataframe
        available_columns = [col for col in selected_columns if col in df.columns]
        
        # Filter the dataframe
        filtered_df = df[available_columns].copy()
        
        # Filter by specific cell if provided
        if cell_id:
            filtered_df = filtered_df[filtered_df['nrCellIdentity'] == cell_id]
            output_suffix += f'_cell{cell_id}'
        
        # Sort by time for chronological analysis
        if 'time' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('time')
        
        # Create output directory if it doesn't exist
        os.makedirs('data/filtered', exist_ok=True)
        
        # Save filtered data
        output_path = f'data/filtered/cell_{output_suffix}.csv'
        filtered_df.to_csv(output_path, index=False)
        
        return f"Filtered data saved to {output_path}. Shape: {filtered_df.shape}. Columns: {list(filtered_df.columns)}"
        
    except Exception as e:
        return f"Error extracting cell data: {str(e)}"

def read_filtered_data(file_path: str) -> dict:
    """
    Reads the filtered CSV data for analysis.
    """
    try:
        df = pd.read_csv(file_path)
        return df.to_dict(orient='records')
    except Exception as e:
        return f"Error reading filtered file {file_path}: {str(e)}"

# Create the data extraction agent
def create_data_extraction_agent(model):
    """
    Creates a specialized data extraction agent that filters data based on query context.
    """
    
    data_extraction_agent = create_react_agent(
        model=model,
        name="data_extraction_agent", 
        prompt=(
            "You are a data extraction specialist for telecom network data. "
            "Your job is to analyze user queries and extract only the relevant data columns and rows. "
            
            "UE DATA COLUMNS EXPLAINED:\n"
            "- time: Timestamp of measurement\n"
            "- ue-id: User Equipment identifier (e.g., 'UE10')\n"
            "- nrCellIdentity: Current serving 5G cell tower ID\n"
            "- DRB.UEThpDl: Downlink throughput performance\n"
            "- RF.serving.RSRP/RSRQ/RSSINR: Signal quality from serving cell\n"
            "- nbCellIdentity_0-3: Neighboring cell identifiers\n"
            "- rsrp_nb0-4, rsrq_nb0-4, rssinr_nb0-4: Signal measurements from neighbors\n"
            "- x, y: UE geographical coordinates\n"
            "- Viavi.UE.anomalies: Detected anomalies count\n"
            "- targetTput: Target throughput\n"
            "- RRU.PrbUsedDl: Physical Resource Blocks used\n"
            
            "CELL DATA COLUMNS EXPLAINED:\n"
            "- time: Timestamp of measurement\n"
            "- nrCellIdentity: Cell tower identifier\n" 
            "- availPrbDl/Ul: Available Physical Resource Blocks (Down/Up-link)\n"
            "- pdcpBytesDl/Ul: Data bytes transmitted (Down/Up-link)\n"
            "- throughput: Cell throughput performance\n"
            "- x, y: Cell geographical coordinates\n"
            
            "When given a query:\n"
            "1. Analyze what data is needed\n"
            "2. Use extract_ue_mobility_data() or extract_cell_performance_data() with the query description\n"
            "3. Include specific IDs (UE ID or Cell ID) if mentioned in the query\n"
            "4. Always return the path to the filtered CSV file\n"
            
            "Examples:\n"
            "- 'Which cell towers has UE10 passed through?' → extract_ue_mobility_data('cell towers passed through', 'UE10')\n"
            "- 'Show signal quality for all UEs' → extract_ue_mobility_data('signal quality')\n"
            "- 'Cell 5 throughput performance' → extract_cell_performance_data('throughput performance', 5)\n"
        ),
        tools=[extract_ue_mobility_data, extract_cell_performance_data, read_filtered_data],
    )
    
    return data_extraction_agent

# Updated network agent that works with filtered data
def create_enhanced_network_agent(model):
    """
    Creates an enhanced network agent that works with pre-filtered data.
    """
    
    enhanced_network_agent = create_react_agent(
        model=model,
        name="enhanced_network_agent",
        prompt=(
            "You are an expert in telecom network performance analysis. "
            "You work with pre-filtered, relevant data provided by the data extraction agent. "
            "When you receive a filtered CSV file path, use read_filtered_data() to analyze it. "
            "Focus on providing specific, accurate insights based on the filtered data. "
            "For mobility queries, track the chronological sequence of cell connections. "
            "For performance queries, analyze trends and identify issues. "
            "Always be precise and base your analysis only on the actual data provided."
        ),
        tools=[read_filtered_data],
    )
    
    return enhanced_network_agent