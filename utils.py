import argparse
from influxdb import InfluxDBClient
import csv
import subprocess
import os

def execute_bash_command(command):
    """Executes a bash command and returns the output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def stream_xapps_logs(num_lines=200):
    log_files = ["trafficxapp.log", "ad.log", "qp.log"]
    for log_file in log_files:
        if os.path.exists(f"data/logs/{log_file}"):
            os.remove(log_file)
            print(f"Deleted old log file: {log_file}")

    ts = "kubectl logs -n ricxapp --tail=200 $(kubectl get pods -n ricxapp -o name | grep ricxapp-trafficxapp) >> data/logs/trafficxapp.log"
    ad = "kubectl logs -n ricxapp --tail=200 $(kubectl get pods -n ricxapp -o name | grep ricxapp-ad) >> data/logs/ad.log"
    qp = "kubectl logs -n ricxapp --tail=200 $(kubectl get pods -n ricxapp -o name | grep ricxapp-qp) >> data/logs/qp.log"
    commands = [ts, ad, qp]
    for command in commands:
        try:
            stdout, stderr, returncode = execute_bash_command(command)
            if returncode != 0:
                print(f"Error executing command '{command}': {stderr}")
            else:
                print(f"Command '{command}' executed successfully.")
        except Exception as e:
            print(f"Exception occurred while executing command '{command}': {e}")


def export_influx_to_csv(host, port, user, password, database, measurement, output_file):
    # Connect to InfluxDB
    client = InfluxDBClient(host=host, username=user, password=password, port=port, database=database)

    # Query data from InfluxDB
    query = f'SELECT * FROM "{measurement}" limit 200'
    result = client.query(query)

    # Write data to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header
        csv_writer.writerow(result.raw['series'][0]['columns'])

        # Write data
        for row in result.raw['series'][0]['values']:
            csv_writer.writerow(row)

    print(f'Data exported to {output_file}')



def get_ue_cell_data(host, password):
    # Placeholder function for future use
    port = 8086
    user = "admin"
    database = "RIC-Test"
    try:
        export_influx_to_csv(host, port, user, password, database, 'CellReports', 'data/kpis/cell.csv')
        export_influx_to_csv(host, port, user, password, database, 'UEReports', 'data/kpis/ue.csv')
        print("UE and Cell data exported successfully.")
    except Exception as e:
        print(f"unable to connect to InfluxDB with host {host} and password {password}. \nError: {e}")
    

def main():
    # InfluxDB connection details
    host = "10.98.44.43"
    port = 8086
    user = "admin"
    password = "JOmxp6DIOD"
    database = "RIC-Test"
    # measurement = "UEReports"
    # output_file = "UE_test.csv"
    parser = argparse.ArgumentParser(description='Export data from InfluxDB to CSV')
    parser.add_argument('--msr', required=True, help='InfluxDB measurement')
    parser.add_argument('--out', required=True, help='Output CSV file name')
    args = parser.parse_args()
    print(args)
    print("start exporting")
    export_influx_to_csv(host, port, user, password, database, args.msr, args.out)
    print("finished")


if __name__ == "__main__":
    main()