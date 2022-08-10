import argparse
import glob 

parser = argparse.ArgumentParser(description='grep top-1 acc, flops and params')
parser.add_argument('-p', '--path_root', help='location of logs.', type=str, nargs="+")

args = parser.parse_args()


def main():
    all_logs = []
    for path in args.path_root:
        all_logs.extend(glob.glob(path + "/**/*.log", recursive=True))
        all_logs += glob.glob(path + "/**/*.log.*", recursive=True)
    all_results = []
    flops = 0
    params = 0
    for log in sorted(all_logs):
        if "scaleup" in log:
            continue

        epoch_expected = log.split('-')[-1].split('/')[0]
        epoch_expected = int(epoch_expected[1:])
        with open(log) as f:
            best_top1 = 0.0
            best_top5 = 0.0
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                if "flops" in line:
                    flops = line.split(": ")[-1]
                elif "Total params:" in line:
                    params = line.split(": ")[-1]
                if "Val" in line:
                    top1 = float(line.split("Top@1: ")[-1][:7])
                    top5 = float(line.split("Top@5: ")[-1][:7])
                    epoch = int(line.split("/")[0][-3:])
                    if top1 > best_top1:
                        best_top1 = top1
                        best_top5 = top5
            try:
                if best_top1 != 0.0:
                    name = log.split("/")[-2]
                    if epoch != epoch_expected:
                        name += '*'
                    all_results.append((name, flops, params, best_top1, best_top5, epoch))
            except:
                pass
    print ("* indicates unfished jobs")            
    print("| Model Name | Top-1 | Top-5 | Flops | Params |Epochs|")
    print("|------------|-------|-------|-------|--------|-------|")
    for name, flop, param, top1, top5, epoch in all_results:
        print("| {} | {} | {} | {} | {} | {}|".format(name, top1, top5, flop, param, epoch))

if __name__ == "__main__":
    main()
