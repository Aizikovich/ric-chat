import argparse
from influxdb import InfluxDBClient
import csv


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