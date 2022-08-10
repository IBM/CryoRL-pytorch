import csv

def saveCTFdict(confidence_file, CTF_file):
    raws = []
    with open(file_name, 'r+') as csvfile:
        files = csv.reader(csvfile)
        for row in files:
            raws.append(row)
    with open(file_output, 'w+') as csvfilew:
        csvwriter = csv.writer(csvfilew)
        for row in raws:
            new_row = row
            new_row.append(score_dict[new_row[0]])
            csvwriter.writerow(new_row)

