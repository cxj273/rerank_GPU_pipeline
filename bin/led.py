#!/usr/bin/env python2.7
import os
import sys
import getopt

def main(argv):
    help_output = "Usage: led.py [-f] <file> <code>"
    err_output = help_output
    try:
        opts, args = getopt.getopt(argv, "f", ["help"])
    except getopt.GetoptError:
        print err_output
        sys.exit(1)

    if len(args) != 2:
        print err_output
        sys.exit(1)
    
    path = args[0]
    code = args[1]
    ifile = open(path, 'r')
    

    is_filter = False
    for o, a in opts:
        if o == '-f':
            is_filter = True
        elif o == '--help':
            print help_output
            sys.exit(0)

    for line in ifile.readlines():
        result = eval_code(code, line)
        if is_filter:
            if result:
                print str(line.rstrip('\n'))
        else: 
            print str(result)

    ifile.close()

def eval_code(code, line):
    return eval(code, {'line':line.rstrip('\n'), 'path':os.path})

if __name__ == "__main__":
    main(sys.argv[1:])
