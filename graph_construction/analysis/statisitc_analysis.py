import pickle 
import csv

with open("repos/bitcoin/bips.pkl", "rb") as f:
    all_rows = pickle.load(f)
    print(f"Found {len(all_rows)} rows in bitcoin/bips.pkl")

# convert all_rows into csv
with open("repos/bitcoin/bips.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(all_rows[0].keys())
    for row in all_rows:
        writer.writerow(row.values())
    print(f"Done writing to bitcoin/bips.csv")


