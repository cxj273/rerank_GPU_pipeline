#!/usr/bin/env python
import sys

def main(argv):
    if len(argv) != 2:
        print "Usage: ./ap.py prediction label"
        sys.exit(1)

    with open(argv[0], 'r') as pred_file:
        with open(argv[1], 'r') as label_file:
            pred_lines = pred_file.readlines()
            score_idx = 1
            if pred_lines[0].split()[1] != "1":
                score_idx = 2

            scores = [float(line.split()[score_idx]) for line in pred_lines[1:]]
            label = [int(line) for line in label_file.readlines()]
            assert len(scores) == len(label)

            ap_list = sorted(zip(label, scores), key = lambda x: x[1], reverse=True)

            ap = 0.0
            pre = 0
            for i in range(len(ap_list)):
                if ap_list[i][0] == 1:
                    pre += 1
                    ap += float(pre) / float(i + 1)

            ap /= pre
            print ap


    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
