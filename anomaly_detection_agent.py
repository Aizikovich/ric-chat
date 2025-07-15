import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
from langgraph.prebuilt import create_react_agent
from datetime import datetime, timedelta

def detect_signal_anomalies(file_path: str, threshold_std: float = 2.0) -> str:
    """
    Detects anomalies in signal quality metrics (RSRP, RSRQ, RSSINR).
    Uses statistical methods to identify outliers.
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        anomalies = []
        
        # Check if signal quality columns exist
        signal_columns = ['RF.serving.RSRP', 'RF.serving.RSRQ', 'RF.serving.RSSINR']
        available_signal_cols = [col for col in signal_columns if col in df.columns]
        
        if not available_signal_cols:
            return "No signal quality columns found in the data."
        
        # Also check for extreme values based on typical ranges
        rsrp_threshold = -100  # dBm - poor signal
        rsrq_threshold = -15   # dB - poor quality
        rssinr_threshold = 0   # dB - poor signal to noise
        
        for col in available_signal_cols:
            if col in df.columns:
                # Calculate statistical measures
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                # Find outliers (values beyond threshold * std deviation)
                lower_bound = mean_val - threshold_std * std_val
                upper_bound = mean_val + threshold_std * std_val
                
                # Statistical outliers
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                # Also check for values below acceptable thresholds
                if col == 'RF.serving.RSRP':
                    poor_signal = df[df[col] < rsrp_threshold]
                    outliers = pd.concat([outliers, poor_signal]).drop_duplicates()
                elif col == 'RF.serving.RSRQ':
                    poor_quality = df[df[col] < rsrq_threshold]
                    outliers = pd.concat([outliers, poor_quality]).drop_duplicates()
                elif col == 'RF.serving.RSSINR':
                    # For RSSINR, convert from large number to dB if needed
                    if df[col].mean() > 1000:  # Looks like raw values
                        df[col + '_dB'] = 10 * np.log10(df[col])
                        poor_sinr = df[df[col + '_dB'] < rssinr_threshold]
                    else:
                        poor_sinr = df[df[col] < rssinr_threshold]
                    outliers = pd.concat([outliers, poor_sinr]).drop_duplicates()
                
                if not outliers.empty:
                    for idx, row in outliers.iterrows():
                        ue_id = row.get('ue-id', 'Unknown')
                        time = row.get('time', 'Unknown')
                        value = row[col]
                        cell = row.get('nrCellIdentity', 'Unknown')
                        
                        # Determine severity based on value
                        if col == 'RF.serving.RSRP':
                            severity = 'Critical' if value < -110 else ('High' if value < -100 else 'Medium')
                        elif col == 'RF.serving.RSRQ':
                            severity = 'Critical' if value < -20 else ('High' if value < -15 else 'Medium')
                        else:
                            severity = 'High' if abs(value - mean_val) > 3 * std_val else 'Medium'
                        
                        anomalies.append({
                            'type': 'Signal Quality Anomaly',
                            'metric': col,
                            'ue_id': ue_id,
                            'time': time,
                            'value': value,
                            'cell': cell,
                            'severity': severity,
                            'description': f"{col} value {value:.2f} is {'poor' if col == 'RF.serving.RSRP' and value < rsrp_threshold else 'significantly below' if value < mean_val else 'above'} normal range [{lower_bound:.2f}, {upper_bound:.2f}]"
                        })
        
        return format_anomaly_report(anomalies, "Signal Quality Anomalies")
        
    except Exception as e:
        return f"Error detecting signal anomalies: {str(e)}"

def detect_throughput_anomalies(file_path: str, min_throughput: float = 10.0) -> str:
    """
    Detects anomalies in throughput performance.
    Identifies sudden drops, zero throughput, and performance degradation.
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        anomalies = []
        
        # Check UE throughput
        if 'DRB.UEThpDl' in df.columns:
            # Sort by time and UE
            if 'time' in df.columns and 'ue-id' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values(['ue-id', 'time'])
                
                for ue in df['ue-id'].unique():
                    ue_data = df[df['ue-id'] == ue]
                    
                    # Check for low throughput
                    low_throughput = ue_data[ue_data['DRB.UEThpDl'] < min_throughput]
                    for idx, row in low_throughput.iterrows():
                        anomalies.append({
                            'type': 'Low Throughput',
                            'metric': 'DRB.UEThpDl',
                            'ue_id': ue,
                            'time': row['time'],
                            'value': row['DRB.UEThpDl'],
                            'cell': row.get('nrCellIdentity', 'Unknown'),
                            'severity': 'Critical' if row['DRB.UEThpDl'] < 1.0 else 'High',
                            'description': f"Throughput {row['DRB.UEThpDl']:.2f} Mbps is below minimum threshold {min_throughput} Mbps"
                        })
                    
                    # Check for sudden drops (more than 50% decrease)
                    if len(ue_data) > 1:
                        ue_data['throughput_change'] = ue_data['DRB.UEThpDl'].pct_change()
                        sudden_drops = ue_data[ue_data['throughput_change'] < -0.5]
                        
                        for idx, row in sudden_drops.iterrows():
                            if idx > 0:  # Skip first row
                                prev_value = ue_data.iloc[idx-1]['DRB.UEThpDl']
                                anomalies.append({
                                    'type': 'Sudden Throughput Drop',
                                    'metric': 'DRB.UEThpDl',
                                    'ue_id': ue,
                                    'time': row['time'],
                                    'value': row['DRB.UEThpDl'],
                                    'cell': row.get('nrCellIdentity', 'Unknown'),
                                    'severity': 'High',
                                    'description': f"Throughput dropped {abs(row['throughput_change']*100):.1f}% from {prev_value:.2f} to {row['DRB.UEThpDl']:.2f} Mbps"
                                })
        
        # Check cell throughput
        elif 'throughput' in df.columns:
            low_cell_throughput = df[df['throughput'] < min_throughput * 10]  # Higher threshold for cells
            for idx, row in low_cell_throughput.iterrows():
                anomalies.append({
                    'type': 'Low Cell Throughput',
                    'metric': 'throughput',
                    'ue_id': 'N/A',
                    'time': row.get('time', 'Unknown'),
                    'value': row['throughput'],
                    'cell': row.get('nrCellIdentity', 'Unknown'),
                    'severity': 'High',
                    'description': f"Cell throughput {row['throughput']:.2f} is critically low"
                })
        
        return format_anomaly_report(anomalies, "Throughput Anomalies")
        
    except Exception as e:
        return f"Error detecting throughput anomalies: {str(e)}"

def detect_handover_anomalies(file_path: str, max_handovers_per_minute: int = 3) -> str:
    """
    Detects anomalies in handover patterns.
    Identifies ping-pong effects, rapid handovers, and unstable connections.
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        anomalies = []
        
        if 'time' in df.columns and 'ue-id' in df.columns and 'nrCellIdentity' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values(['ue-id', 'time'])
            
            for ue in df['ue-id'].unique():
                ue_data = df[df['ue-id'] == ue].copy()
                
                # Detect cell changes
                ue_data['cell_changed'] = ue_data['nrCellIdentity'].ne(ue_data['nrCellIdentity'].shift())
                handovers = ue_data[ue_data['cell_changed']].copy()  # Use copy() to avoid warning
                
                if len(handovers) > 1:
                    # Check for rapid handovers
                    handovers.loc[:, 'time_diff'] = handovers['time'].diff()  # Use loc to avoid warning
                    rapid_handovers = handovers[handovers['time_diff'] < timedelta(seconds=20)]
                    
                    for idx, row in rapid_handovers.iterrows():
                        if pd.notna(row['time_diff']):
                            anomalies.append({
                                'type': 'Rapid Handover',
                                'metric': 'Handover Rate',
                                'ue_id': ue,
                                'time': row['time'],
                                'value': row['time_diff'].total_seconds(),
                                'cell': row['nrCellIdentity'],
                                'severity': 'High',
                                'description': f"Handover occurred only {row['time_diff'].total_seconds():.1f} seconds after previous handover"
                            })
                    
                    # Check for ping-pong effect (returning to previous cell)
                    for i in range(2, len(handovers)):
                        if handovers.iloc[i]['nrCellIdentity'] == handovers.iloc[i-2]['nrCellIdentity']:
                            time_window = (handovers.iloc[i]['time'] - handovers.iloc[i-2]['time']).total_seconds()
                            if time_window < 60:  # Within 1 minute
                                anomalies.append({
                                    'type': 'Ping-Pong Handover',
                                    'metric': 'Handover Pattern',
                                    'ue_id': ue,
                                    'time': handovers.iloc[i]['time'],
                                    'value': time_window,
                                    'cell': handovers.iloc[i]['nrCellIdentity'],
                                    'severity': 'Critical',
                                    'description': f"UE returned to cell {handovers.iloc[i]['nrCellIdentity']} within {time_window:.1f} seconds (ping-pong effect)"
                                })
        
        return format_anomaly_report(anomalies, "Handover Anomalies")
        
    except Exception as e:
        return f"Error detecting handover anomalies: {str(e)}"

def detect_resource_anomalies(file_path: str, prb_threshold: float = 90.0) -> str:
    """
    Detects anomalies in resource usage (PRB utilization).
    Identifies resource exhaustion and abnormal usage patterns.
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        anomalies = []
        
        # Check PRB usage for UEs
        if 'RRU.PrbUsedDl' in df.columns:
            high_prb_usage = df[df['RRU.PrbUsedDl'] > prb_threshold]
            for idx, row in high_prb_usage.iterrows():
                anomalies.append({
                    'type': 'High PRB Usage',
                    'metric': 'RRU.PrbUsedDl',
                    'ue_id': row.get('ue-id', 'Unknown'),
                    'time': row.get('time', 'Unknown'),
                    'value': row['RRU.PrbUsedDl'],
                    'cell': row.get('nrCellIdentity', 'Unknown'),
                    'severity': 'High',
                    'description': f"PRB usage {row['RRU.PrbUsedDl']:.1f}% exceeds threshold {prb_threshold}%"
                })
        
        # Check available PRB for cells
        if 'availPrbDl' in df.columns:
            low_avail_prb = df[df['availPrbDl'] < 10]
            for idx, row in low_avail_prb.iterrows():
                anomalies.append({
                    'type': 'Low Available PRB',
                    'metric': 'availPrbDl',
                    'ue_id': 'N/A',
                    'time': row.get('time', 'Unknown'),
                    'value': row['availPrbDl'],
                    'cell': row.get('nrCellIdentity', 'Unknown'),
                    'severity': 'Critical',
                    'description': f"Only {row['availPrbDl']} PRBs available - cell resources nearly exhausted"
                })
        
        return format_anomaly_report(anomalies, "Resource Usage Anomalies")
        
    except Exception as e:
        return f"Error detecting resource anomalies: {str(e)}"

def detect_comprehensive_anomalies(file_path: str) -> str:
    """
    Runs all anomaly detection methods and provides a comprehensive report.
    """
    try:
        # First, get data statistics
        stats = analyze_data_statistics(file_path)
        
        all_results = [stats]
        
        # Run all detection methods
        signal_anomalies = detect_signal_anomalies(file_path)
        throughput_anomalies = detect_throughput_anomalies(file_path)
        handover_anomalies = detect_handover_anomalies(file_path)
        resource_anomalies = detect_resource_anomalies(file_path)
        
        # Check for existing anomaly column
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        if 'Viavi.UE.anomalies' in df.columns:
            reported_anomalies = df[df['Viavi.UE.anomalies'] > 0]
            if not reported_anomalies.empty:
                viavi_report = f"\n=== System-Reported Anomalies (Viavi) ===\n"
                viavi_report += f"Found {len(reported_anomalies)} entries with Viavi anomalies:\n"
                for idx, row in reported_anomalies.head(10).iterrows():  # Show up to 10
                    viavi_report += f"- UE: {row.get('ue-id', 'Unknown')}, Time: {row.get('time', 'Unknown')}, Count: {row['Viavi.UE.anomalies']}, Cell: {row.get('nrCellIdentity', 'Unknown')}\n"
                all_results.append(viavi_report)
        
        # Combine all results
        all_results.extend([signal_anomalies, throughput_anomalies, handover_anomalies, resource_anomalies])
        
        # Create summary
        summary = "\n=== COMPREHENSIVE ANOMALY DETECTION REPORT ===\n"
        summary += f"Analysis performed on: {file_path}\n"
        summary += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += "\n" + "\n".join(all_results)
        
        return summary
        
    except Exception as e:
        return f"Error in comprehensive anomaly detection: {str(e)}"

def analyze_data_statistics(file_path: str) -> str:
    """
    Provides statistics about the data to help understand why anomalies might not be detected.
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        stats = f"\n=== Data Statistics for {file_path} ===\n"
        stats += f"Total rows: {len(df)}\n"
        
        if 'ue-id' in df.columns:
            stats += f"Unique UEs: {df['ue-id'].nunique()}\n"
            stats += f"UE IDs: {df['ue-id'].unique()}\n"
        
        if 'nrCellIdentity' in df.columns:
            stats += f"Unique cells: {df['nrCellIdentity'].nunique()}\n"
            stats += f"Cells: {df['nrCellIdentity'].unique()}\n"
        
        # Signal statistics
        for col in ['RF.serving.RSRP', 'RF.serving.RSRQ', 'RF.serving.RSSINR']:
            if col in df.columns:
                stats += f"\n{col}:\n"
                stats += f"  Mean: {df[col].mean():.2f}\n"
                stats += f"  Std: {df[col].std():.2f}\n"
                stats += f"  Min: {df[col].min():.2f}\n"
                stats += f"  Max: {df[col].max():.2f}\n"
        
        # Throughput statistics
        if 'DRB.UEThpDl' in df.columns:
            stats += f"\nDRB.UEThpDl (Throughput):\n"
            stats += f"  Mean: {df['DRB.UEThpDl'].mean():.2f} Mbps\n"
            stats += f"  Min: {df['DRB.UEThpDl'].min():.2f} Mbps\n"
            stats += f"  Max: {df['DRB.UEThpDl'].max():.2f} Mbps\n"
        
        return stats
        
    except Exception as e:
        return f"Error analyzing data statistics: {str(e)}"
    """
    Formats anomaly findings into a readable report.
    """
    if not anomalies:
        return f"\n=== {title} ===\nNo anomalies detected.\n"
    
    report = f"\n=== {title} ===\n"
    report += f"Found {len(anomalies)} anomalies:\n\n"
    
    # Group by severity
    critical = [a for a in anomalies if a.get('severity') == 'Critical']
    high = [a for a in anomalies if a.get('severity') == 'High']
    medium = [a for a in anomalies if a.get('severity') == 'Medium']
    
    if critical:
        report += f"CRITICAL ({len(critical)}):\n"
        for a in critical[:5]:  # Show first 5
            report += f"  - [{a['type']}] UE: {a['ue_id']}, Cell: {a['cell']}, Time: {a['time']}\n"
            report += f"    {a['description']}\n"
    
    if high:
        report += f"\nHIGH ({len(high)}):\n"
        for a in high[:5]:  # Show first 5
            report += f"  - [{a['type']}] UE: {a['ue_id']}, Cell: {a['cell']}, Time: {a['time']}\n"
            report += f"    {a['description']}\n"
    
    if medium:
        report += f"\nMEDIUM ({len(medium)}):\n"
        for a in medium[:3]:  # Show first 3
            report += f"  - [{a['type']}] UE: {a['ue_id']}, Cell: {a['cell']}\n"
            report += f"    {a['description']}\n"
    
    return report

def create_anomaly_detection_agent(model):
    """
    Creates an anomaly detection agent specialized in identifying network issues.
    """
    
    anomaly_agent = create_react_agent(
        model=model,
        name="anomaly_detection_agent",
        prompt=(
            "You are an expert anomaly detection specialist for telecom networks. "
            "Your role is to analyze network data and identify various types of anomalies. "
            
            "IMPORTANT: You must ALWAYS use detect_comprehensive_anomalies() for general anomaly detection requests!\n"
            "This function runs ALL detection methods including:\n"
            "- Signal quality analysis (RSRP, RSRQ, RSSINR)\n"
            "- Throughput performance issues\n"
            "- Handover problems (ping-pong, rapid handovers)\n"
            "- Resource usage anomalies\n"
            "- System-reported anomalies (Viavi)\n"
            
            "Available detection functions:\n"
            "1. detect_comprehensive_anomalies(file_path) - USE THIS FOR ALL GENERAL ANOMALY REQUESTS\n"
            "2. detect_signal_anomalies(file_path) - Only for specific signal analysis\n"
            "3. detect_throughput_anomalies(file_path) - Only for specific throughput analysis\n"
            "4. detect_handover_anomalies(file_path) - Only for specific handover analysis\n"
            "5. detect_resource_anomalies(file_path) - Only for specific resource analysis\n"
            
            "When you receive a file path:\n"
            "1. If asked to 'detect anomalies' or 'find issues' → use detect_comprehensive_anomalies()\n"
            "2. Only use specific functions if explicitly asked for that type\n"
            "3. The file path will be something like 'data/filtered/ue_anomaly_analysis_UE5.csv'\n"
            "4. Always interpret and summarize the results\n"
            "5. Provide actionable recommendations\n"
            
            "The comprehensive analysis will:\n"
            "- Use statistical methods to find outliers in signal quality\n"
            "- Detect sudden throughput drops and low performance\n"
            "- Identify rapid handovers and ping-pong effects\n"
            "- Find resource exhaustion issues\n"
            "- Report system-detected anomalies\n"
            
            "Remember: For any general anomaly detection request, ALWAYS use detect_comprehensive_anomalies()!"
        ),
        tools=[
            detect_signal_anomalies,
            detect_throughput_anomalies,
            detect_handover_anomalies,
            detect_resource_anomalies,
            detect_comprehensive_anomalies,
            analyze_data_statistics
        ],
    )
    
    return anomaly_agent