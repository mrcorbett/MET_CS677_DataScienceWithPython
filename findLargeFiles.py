
 #! /usr/bin/env python3
# -*- coding: utf-8 -*-
# findlargefiles.py Searches a file location and subdirectories for
# files larger than a given size.
# 
# https://codereview.stackexchange.com/questions/174754/finding-large-files

import argparse
import os


def search_folder(location, min_filesize):
    min_filesize = min_filesize * 1024 ** 2

    count_largefiles = 0
    for foldername, subfolders, filenames in os.walk(location):
        for filename in filenames:
            try:
                size_bytes = os.path.getsize(os.path.join(foldername, filename))
                if min_filesize <= size_bytes:
                    count_largefiles += 1
                    yield foldername + '/' + filename, size_bytes/(1024**2)
            except FileNotFoundError:
                # maybe log error, maybe `pass`, maybe raise an exception
                # (halting further processing), maybe return an error object
                print('EXCEPTION:  File not found')
    if count_largefiles > 0:
        print('Number of large files: ', str(count_largefiles))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find files larger than specified sizes.')
    parser.add_argument('filesize', type=int, action='store', help='Filesize in megabytes')
    parser.add_argument('location', type=str, action='store', help='The directory to search')
    args = parser.parse_args()

    for filename, size in search_folder(args.location, args.filesize):
        print('{:.02f} MB  - {} '.format(size, filename))