from helpers import export_dataset
import argparse

def main(dump, to_file, max_seq=250000, min_th=5):
    print "Exporting to %s" % to_file
    export_dataset(dump, to_file, max_seq, min_th)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dump',help="Dump file to export")
    parser.add_argument('file', help="Path to dump the file to")
    parser.add_argument('--seq', default=250000, help="Maximum sequence size", type=int)
    parser.add_argument('--min_th', default=5, help="Minimum threshold", type=int)
    args = parser.parse_args()
    main(args.dump, args.file, args.seq, args.min_th)