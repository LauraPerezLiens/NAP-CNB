import os
import csv
import requests
from pathlib import Path

DATA_RAW = Path("data_raw")
DATA_INTERMEDIATE = Path("data_intermediate")
DATA_INTERMEDIATE.mkdir(exist_ok=True)

# Create subfolders for mouse and human
for species in ["mouse", "human"]:
    (DATA_INTERMEDIATE / species).mkdir(exist_ok=True)

MAX_ROWS = 1000  # Limit to first 1000 rows per file

# Iterate over all CSV files in the data_raw structure
for species_dir in DATA_RAW.iterdir():
    if not species_dir.is_dir():
        continue
    for mode_dir in species_dir.iterdir():
        for class_dir in mode_dir.iterdir():
            for haplo_dir in class_dir.iterdir():
                csv_file = haplo_dir / "export.csv"
                if not csv_file.exists():
                    continue
                haplo_name = haplo_dir.name
                # Output file in data_intermediate/species
                output_file = DATA_INTERMEDIATE / species_dir.name / f"{mode_dir.name}_{class_dir.name}_{haplo_name}_proteins.csv"
                print(f"Processing {csv_file} -> {output_file}")
                # Read protein_id and count occurrences
                protein_counts = {}
                row_count = 0
                with open(csv_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row_count >= MAX_ROWS:
                            break
                        pid = row.get("protein_id", "")
                        if pid:
                            protein_counts[pid] = protein_counts.get(pid, 0) + 1
                        row_count += 1
                # Download sequences and write result
                with open(output_file, "w", newline="") as out:
                    writer = csv.writer(out)
                    writer.writerow(["count", "protein_id", "aas"])
                    for i, (pid, count) in enumerate(protein_counts.items(), 1):
                        seq = ""
                        if pid.startswith("http"):
                            try:
                                r = requests.get(f"{pid}.fasta", timeout=10)
                                if r.ok:
                                    lines = r.text.splitlines()
                                    seq = "".join(lines[1:])
                            except Exception as e:
                                print(f"Error downloading {pid}: {e}")
                        writer.writerow([count, pid, seq])
                        if i % 200 == 0:
                            print(f"Processed {i} proteins for {haplo_name}")
print("All protein sequence files have been generated successfully.")

def get_unique_protein_ids(merged_files):
    protein_ids = set()
    for file in merged_files:
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get('protein_id')
                if pid:
                    protein_ids.add(pid)
    return protein_ids

def fetch_fasta_uniprot(protein_id):
    url = f"https://www.uniprot.org/uniprot/{protein_id}.fasta"
    resp = requests.get(url)
    if resp.status_code == 200 and resp.text.startswith('>'):
        return resp.text
    return None

def save_fasta_csv(protein_ids, out_csv):
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['protein_id', 'fasta'])
        for pid in protein_ids:
            fasta = fetch_fasta_uniprot(pid)
            if fasta:
                writer.writerow([pid, fasta])

if __name__ == "__main__":
    base_dir = Path("/home/nap/lperez_nn/data/data_raw")
    interm_dir = Path("/home/nap/lperez_nn/data/data_intermediate")
    interm_dir.mkdir(exist_ok=True)

    # Humanos
    human_files = list(base_dir.glob("human/*/*/merged_tcell%mhc.csv"))
    human_proteins = get_unique_protein_ids(human_files)
    save_fasta_csv(human_proteins, interm_dir / "fasta_human.csv")

    # Ratones
    mouse_files = list(base_dir.glob("mouse/*/*/merged_tcell%mhc.csv"))
    mouse_proteins = get_unique_protein_ids(mouse_files)
    save_fasta_csv(mouse_proteins, interm_dir / "fasta_mouse.csv")
