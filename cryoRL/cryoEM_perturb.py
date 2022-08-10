import csv
import random
import numpy as np

def clip(my_value, min_value, max_value):
    return max(min(my_value, max_value), min_value)


def perturb_data(CTF_FILE, error_rate, error_mean, error_std, CTF_threshold, min_value, max_value):
    with open(CTF_FILE, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        ctfs = [row for row in csvreader]
        # for i in ctfs:
            # print(i)

    total_num = len(ctfs)
    error_count = 0
    r = np.random.permutation(total_num)
   # for i,row in enumerate(ctfs):
    for i in r:
        row = ctfs[i]
        CTF = float(row[3])
        lowCTFflag = CTF <= CTF_threshold
        if error_count < total_num * error_rate:
            delta = random.gauss(error_mean, error_std)
            CTF_pred = CTF + delta
            CTF_pred = clip(CTF_pred, min_value, max_value)
            if (CTF_pred <= CTF_threshold) == lowCTFflag:
                CTF_pred = CTF - delta
                CTF_pred = clip(CTF_pred, min_value, max_value)

            if (CTF_pred <= CTF_threshold) != lowCTFflag:
                error_count += 1

            ctfs[i].append(CTF_pred)
        else:
            while True:
                CTF_pred = CTF + random.gauss(error_mean, error_std)
                CTF_pred = clip(CTF_pred, min_value, max_value)
                if (CTF_pred <= CTF_threshold) == lowCTFflag:
                    ctfs[i].append(CTF_pred)
                    break
    errors = [(float(row[3]) <= CTF_threshold) != (row[4] <= CTF_threshold) for row in ctfs]
    print('error rate:',sum(errors)*1.0/len(errors))

    with open(CTF_FILE[:-4]+'_ErrorRate'+str(int(error_rate*100))+'_ErrorMean'+str(error_mean)+'_ErrorStd'+str(error_std)+'.csv', 'w+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(ctfs)

if __name__ == '__main__':
    err_rate = 0.3
    err_mean = 0
    err_std = 2
    ctf_thresh = 6.0
    min_val = 2.0
    max_val = 20.0
    perturb_data('CryoEM_data/target_CTF_A.csv',
                 error_rate=err_rate,
                 error_mean=err_mean,
                 error_std=err_std,
                 CTF_threshold=ctf_thresh,
                 min_value=min_val,
                 max_value=max_val)
