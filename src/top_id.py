#!/usr/bin/env python

import sys

def main(argv):
    if len(argv) != 2:
        print "Usage: ./top_id.py prediction evl_list"
        sys.exit(1)

    pred_file = open(argv[0] , 'r')
    evl_file = open(argv[1], 'r')

    prediction = pred_file.readlines()
    evl = evl_file.readlines()

    if prediction[0].split()[1] == '1':
        evl_pos = 1
    else:
        evl_pos = 2
    
    combined = zip( [float(x.split()[evl_pos]) for x in prediction[1:]], [x.rstrip('\n') for x in evl])

    for x in sorted(combined, key=lambda x: x[0], reverse=True)[:20]:
        print x[1]
    pred_file.close()
    evl_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
