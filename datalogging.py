import csv
import os

def writeResultsCSV(results, file_name, file_location):

    filepath = os.path.join(file_location, f"{file_name}.csv")
    os.makedirs(file_location, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = results[0].keys())
        writer.writeheader()
        writer.writerows(results)
