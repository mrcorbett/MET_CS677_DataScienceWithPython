import argparse
from nbconvert import PythonExporter
import nbformat
import os

# https://nbconvert.readthedocs.io/en/latest/usage.html

#ipython nbconvert -- to script './bicycle_crashes_neiss_v7.ipynb'

#jupyter nbconvert --to PDF './bicycle_crashes_neiss_v7.ipynb'

def convertNotebook(notebookPath, modulePath):
    with open(notebookPath) as fh:
        nb = nbformat.reads(fh.read(), nbformat.NO_CONVERT)

    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(nb)

    with open(modulePath, 'w+') as fh:
        #fh.writelines(source.encode('utf-8'))
        fh.writelines(source)
        print('Wrote', modulePath)

def main(args):
    if args.fname:
        fname_ipynb = args.fname
        fname_py = args.fname
        pre, ext = os.path.splitext(fname_py)
        fname_py = pre + '.py'
        convertNotebook(fname_ipynb, fname_py)
    else:
        raise Exception('Expected an ipynb filename')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('fname', help='The ipynb filename to translate')

    args = parser.parse_args()
    main(args)

