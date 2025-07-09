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
        # Check if the source file exists
        source_file = 'data/kpis/ue_0_100_seed_60.csv'
        if not os.path.exists(source_file):
            return f"Error: Source file {source_file} not found. Please ensure data files are present."
            
        # Read the full UE dataset
        df = pd.read_csv(source_file)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # If empty dataframe, return early
        if df.empty:
            return "Error: UE dataset is empty"
        
        # Define column mappings for different query types
        mobility_columns = ['time', 'ue-id', 'nrCellIdentity', 'x', 'y', 'step']
        signal_quality_columns = ['time', 'ue-id', 'nrCellIdentity', 'RF.serving.RSRP', 'RF.serving.RSRQ', 'RF.serving.RSSINR']
        neighbor_cell_columns = ['time', 'ue-id', 'nrCellIdentity'] + [f'nbCellIdentity_{i}' for i in range(4)] + [f'rsrp_nb{i}' for i in range(5)]
        throughput_columns = ['time', 'ue-id', 'nrCellIdentity', 'DRB.UEThpDl', 'targetTput', 'RRU.PrbUsedDl']
        anomaly_columns = ['time', 'ue-id', 'nrCellIdentity', 'Viavi.UE.anomalies', 'x', 'y']
        
        # Determine relevant columns based on query description
        query_lower = query_description.lower()
        
        if any(keyword in query_lower for keyword in ['cell tower', 'cells', 'passed through', 'mobility', 'movement', 'handover', 'travelled']):
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
        
        if not available_columns:
            return f"Error: None of the required columns {selected_columns} exist in the dataset. Available columns: {list(df.columns)}"
        
        # Filter the dataframe
        filtered_df = df[available_columns].copy()
        
        # Filter by specific UE if provided
        if ue_id:
            # Strip whitespace from ue-id column values
            if 'ue-id' in filtered_df.columns:
                filtered_df['ue-id'] = filtered_df['ue-id'].str.strip()
                df['ue-id'] = df['ue-id'].str.strip()  # Also strip from original df for the error message
            
            # Normalize UE ID format (handle different cases like 'UE10', 'ue10', '10')
            ue_id_normalized = ue_id.upper() if not ue_id.upper().startswith('UE') else ue_id.upper()
            if not ue_id_normalized.startswith('UE'):
                ue_id_normalized = f'UE{ue_id_normalized}'
                
            filtered_df = filtered_df[filtered_df['ue-id'].str.upper() == ue_id_normalized]
            output_suffix += f'_{ue_id_normalized}'
            
            if filtered_df.empty:
                return f"No data found for UE ID: {ue_id}. Available UE IDs: {sorted(df['ue-id'].unique())}"
        
        # Sort by time for chronological analysis
        if 'time' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('time')
        
        # Create output directory if it doesn't exist
        os.makedirs('data/filtered', exist_ok=True)
        
        # Save filtered data
        output_path = f'data/filtered/ue_{output_suffix}.csv'
        filtered_df.to_csv(output_path, index=False)
        
        # For mobility queries, include unique cells visited
        if output_suffix == 'mobility' and 'nrCellIdentity' in filtered_df.columns:
            unique_cells = filtered_df['nrCellIdentity'].unique()
            cells_info = f" Unique cells visited: {list(unique_cells)}"
        else:
            cells_info = ""
        
        return f"Success: Filtered data saved to {output_path}. Shape: {filtered_df.shape}. Columns: {list(filtered_df.columns)}.{cells_info}"
        
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
        # Check if the source file exists
        source_file = 'data/kpis/cell.csv'
        if not os.path.exists(source_file):
            return f"Error: Source file {source_file} not found. Please ensure data files are present."
            
        # Read the full cell dataset
        df = pd.read_csv(source_file)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # If empty dataframe, return early
        if df.empty:
            return "Error: Cell dataset is empty"
        
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
        
        if not available_columns:
            return f"Error: None of the required columns {selected_columns} exist in the dataset. Available columns: {list(df.columns)}"
        
        # Filter the dataframe
        filtered_df = df[available_columns].copy()
        
        # Filter by specific cell if provided
        if cell_id is not None:
            filtered_df = filtered_df[filtered_df['nrCellIdentity'] == cell_id]
            output_suffix += f'_cell{cell_id}'
            
            if filtered_df.empty:
                return f"No data found for Cell ID: {cell_id}. Available Cell IDs: {df['nrCellIdentity'].unique()[:10]}..."
        
        # Sort by time for chronological analysis
        if 'time' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('time')
        
        # Create output directory if it doesn't exist
        os.makedirs('data/filtered', exist_ok=True)
        
        # Save filtered data
        output_path = f'data/filtered/cell_{output_suffix}.csv'
        filtered_df.to_csv(output_path, index=False)
        
        return f"Success: Filtered data saved to {output_path}. Shape: {filtered_df.shape}. Columns: {list(filtered_df.columns)}"
        
    except Exception as e:
        return f"Error extracting cell data: {str(e)}"

def read_filtered_data(file_path: str) -> str:
    """
    Reads the filtered CSV data for analysis.
    Returns a string summary instead of dict to avoid recursion issues.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File {file_path} not found"
            
        df = pd.read_csv(file_path)
        
        # Strip whitespace from column names (in case of formatting issues)
        df.columns = df.columns.str.strip()
        
        # Create a summary instead of returning all data
        summary = f"File: {file_path}\n"
        summary += f"Shape: {df.shape}\n"
        summary += f"Columns: {list(df.columns)}\n"
        
        # Add specific summaries based on data type
        if 'ue-id' in df.columns and 'nrCellIdentity' in df.columns:
            # UE mobility data
            if 'nrCellIdentity' in df.columns:
                unique_cells = df['nrCellIdentity'].unique()
                summary += f"\nUnique cells: {list(unique_cells)}\n"
                summary += f"Number of unique cells: {len(unique_cells)}\n"
                
                # Cell visit sequence for each UE
                for ue in df['ue-id'].unique():
                    ue_data = df[df['ue-id'] == ue]
                    cell_sequence = ue_data['nrCellIdentity'].tolist()
                    summary += f"\n{ue} cell sequence: {cell_sequence[:20]}{'...' if len(cell_sequence) > 20 else ''}"
                    
        elif 'nrCellIdentity' in df.columns and 'throughput' in df.columns:
            # Cell performance data
            summary += f"\nCell IDs: {df['nrCellIdentity'].unique()}\n"
            if 'throughput' in df.columns:
                summary += f"Throughput range: {df['throughput'].min():.2f} - {df['throughput'].max():.2f}\n"
        
        # Add first few rows as sample
        summary += f"\nFirst 5 rows:\n{df.head().to_string()}"
        
        return summary
        
    except Exception as e:
        return f"Error reading filtered file {file_path}: {str(e)}"

# Create the data extraction agent with improved error handling
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
            
            "IMPORTANT: You must complete your task in ONE attempt. Do not loop or retry.\n"
            "If a function returns an error, report it and stop.\n"
            
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
            "4. Return the result immediately - do not retry or loop\n"
            "5. If the extraction was successful, return the file path. If it failed, return the error message.\n"
            
            "Examples:\n"
            "- 'Which cell towers has UE10 passed through?' → extract_ue_mobility_data('cell towers passed through travelled', 'UE10')\n"
            "- 'Show signal quality for all UEs' → extract_ue_mobility_data('signal quality')\n"
            "- 'Cell 5 throughput performance' → extract_cell_performance_data('throughput performance', 5)\n"
            
            "REMEMBER: Execute ONE function call and return the result. Do not loop."
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
            "The function will return a summary of the data including unique cells visited. "
            "Focus on providing specific, accurate insights based on the filtered data. "
            "For mobility queries, report the chronological sequence of cell connections. "
            "For performance queries, analyze trends and identify issues. "
            "Always be precise and base your analysis only on the actual data provided."
        ),
        tools=[read_filtered_data],
    )
    
    return enhanced_network_agent