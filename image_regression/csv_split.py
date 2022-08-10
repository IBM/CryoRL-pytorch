import os
import argparse
import pandas as pd

def setupParserOptions():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imagedir', default='')
    ap.add_argument('--ctffile', default='target_CTF_M.csv')
    ap.add_argument('--outname', default='')
    args = vars(ap.parse_args())
    return args


def main():

    args = setupParserOptions()
    df = pd.read_csv(args['ctffile'], header=None, index_col=0)
    img_list = []

    for img in os.listdir(args['imagedir']):
        img_list.append(img[:-9])

    df2 = df.loc[img_list]
    df2.to_csv(args['outname'], header=False)

if __name__ == '__main__':
    main()
