import numpy as np
import csv
def get_img_names(PREDICTION_FILE_NAME):
    data = np.loadtxt(PREDICTION_FILE_NAME, delimiter=' ', dtype=str)
    return [p[0] for p in data]

def split_by_name_ts(FILE_NAME, Img_names):
    with open(FILE_NAME, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        return [(row[0],row[1]) for row in csvreader if row[0] in Img_names]
    
def split_by_name_ctf(FILE_NAME, Img_names):
    with open(FILE_NAME, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        return [(row[0],row[3], row[4]) for row in csvreader if row[0] in Img_names]