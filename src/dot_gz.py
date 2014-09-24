#!/usr/bin/env python
import sys
import gzip

def main(argv):
    if len(argv) != 3:
        print "Usage: ./dot_gz.py dim a b"
        sys.exit(1)

    dim = int(argv[0])
    with gzip.open(argv[1]) as a_file:
        with gzip.open(argv[2]) as b_file:
            a = [0] * dim
            b = [0] * dim
            for item in a_file.read().split():
                idx = int(item.split(':')[0]) - 1 
                value = float(item.split(':')[1]) 
                a[idx] = value

            for item in b_file.read().split(): 
                idx = int(item.split(':')[0]) - 1 
                value = float(item.split(':')[1]) 
                b[idx] = value
            
            print sum([item[0]*item[1] for item in zip(a,b)])
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
