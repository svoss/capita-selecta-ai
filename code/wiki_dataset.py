import os
import sys
import numpy as np
import logging
import hashlib
import fileinput
from chainer.dataset import download
import shutil
import tempfile
import re
import codecs
import requests
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(),__file__)))
GITHUB_ZIP = "https://github.com/attardi/wikiextractor/archive/master.zip"


def install_wiki_extractor():
    # if wikiextractor folder does not exist
    wiki_extractor_path = os.path.join(ROOT, 'wikiextractor')
    if not os.path.isdir(wiki_extractor_path):
        # download zip from repo
        print "Downloading wiki extractor"
        r = requests.get(GITHUB_ZIP)
        zip_path = os.path.join(ROOT, 'wikiextractor.zip')
        with open(zip_path, "wb") as code:
            code.write(r.content)

        # extract zip from zip
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(ROOT)

        # by default the zip is extract to ROOT/wikiextractor-master we move this to the ROOT/wikiextractor path
        shutil.move(os.path.join(ROOT, 'wikiextractor-master'), wiki_extractor_path)

        # remove zip file
        os.remove(zip_path)
        # append to path
    sys.path.append(wiki_extractor_path)


install_wiki_extractor()
from WikiExtractor import Extractor, minFileSize, ignoreTag, load_templates, pages_from,process_dump,cpu_count,filter_disambig_pages
#very ugly hack, to make discardElements global work
import __builtin__
__builtin__.discardElements = []


def get_wiki_dataset(url, max=-1):
    """
    Gets sequence dataset of wikipedia dump. Retrieved from here: https://dumps.wikimedia.org/backup-index.html
    Will download the dataset, extract the content using WikiExtractor(https://github.com/attardi/wikiextractor-2) to extract content from dump
    Next it will tokenize text and build a sequence array: a list of integers that represent word sequences and a vocabulary
    where the index of each element will be the integer used in the sequence of that word in the seq array
    Dataset is not shuffled and order of sentences will be respected on an article basis, "." will be replaces by <eos> tokens

    :param: url link to the dump on wikipedia
    :param: Limit number of tokens in sequence
    :return: seq, voc
    """

    def creator(path):
        dump_path = download.cached_download(url)
        tmp_dir = tempfile.mkdtemp()

        # WikiExtractor needs .bz2 extension to function well
        dump_sym = os.path.join(tmp_dir, 'dump.bz2')
        os.symlink(dump_path, dump_sym)
        print "Extracting dump..."

        extract_dir = os.path.join(tmp_dir,'extracts')
        extract_dump(dump_sym, extract_dir, quiet=True)

        print "Building vocabulary and sequence array.."
        seq,voc = _build_dataset(extract_dir, path,max)

        # clean up temp file:
        print "Removing dump"
        shutil.rmtree(extract_dir)

        return seq, voc

    def loader(path):
        with open(path) as io:
            data = np.load(io)
            return data['seq'],data['voc']



    root = download.get_dataset_directory('svoss/chainer/wiki')
    path = os.path.join(root, hashlib.md5(url).hexdigest()+("_%d" % max)+".npz")
    return download.cache_or_load_file(path, creator, loader)

def _build_dataset(extract_dir, target_path, max):
    print max
    seq = []
    count = 0
    words = {}# word => index, for fast index retrieval
    word_list = []
    last_index = 0
    for current,dirs,files in os.walk(extract_dir):
        for file in files:
            if file.startswith('wiki_'):
                f = os.path.join(current, file)
                with codecs.open(f,'r',encoding='utf8') as io:
                    for line in io:
                        # This regex matches <doc>  and </doc> tags in the generated files, which should be ignored
                        if re.match(r"\<\/?doc(.*)\>",line) is None:
                            # removes <br> and replace . with <eos>
                            line = line.replace("<br>"," ").replace("."," <eos>")
                            for token in re.findall("[\w\<\>]+",line):
                                if token not in words:
                                    words[token] = last_index
                                    word_list.append(token)
                                    last_index += 1
                                count += 1
                                seq.append(words[token])
                        if count > max and max > -1:
                            break
            if count > max and max > -1:
                break
        if count > max and max > -1:
            break

    seq = np.array(seq, dtype=np.uint32)
    words = np.array(word_list, dtype=np.dtype('str'))
    with open(target_path,'w') as io:
        np.savez(io, seq=seq, voc=words)
    return seq, words


# function extracts templates from dump, it uses the WikiExtractor module to do so
# Based on the main() function in WikiExtractor.py, but replaces params
def extract_dump(input, output=None, bytes="1M", compress=False, html=False, links=False, sections=False, lists=False,
                 namespaces=False, templates=False, no_templates=True, revision=False,
                 min_text_length=Extractor.min_text_length, arg_filter_disambig_pages=False, processes=False, quiet=False,
                 debug=False, article=False, version="",discard_elements=False):
    global urlbase, acceptedNamespaces, filter_disambig_pages
    global templateCache
    global discardElements

    if discard_elements:
        discardElements = set(discard_elements.split(','))
    else:
        discardElements = [
            'gallery', 'timeline', 'noinclude', 'pre',
            'table', 'tr', 'td', 'th', 'caption', 'div',
            'form', 'input', 'select', 'option', 'textarea',
            'ul', 'li', 'ol', 'dl', 'dt', 'dd', 'menu', 'dir',
            'ref', 'references', 'img', 'imagemap', 'source', 'small',
            'sub', 'sup', 'indicator'
        ]


    default_process_count = cpu_count() - 1
    if processes is False:
        processes = default_process_count

    Extractor.keepLinks = links
    Extractor.keepSections = sections
    Extractor.keepLists = lists
    Extractor.toHTML = html
    Extractor.print_revision = revision
    Extractor.min_text_length = min_text_length
    if html:
        Extractor.keepLinks = True

    Extractor.expand_templates = no_templates
    filter_disambig_pages = arg_filter_disambig_pages

    try:
        power = 'kmg'.find(bytes[-1].lower()) + 1
        file_size = int(bytes[:-1]) * 1024 ** power
        if file_size < minFileSize:
            raise ValueError()
    except ValueError:
        logging.error('Insufficient or invalid size: %s', bytes)
        return

    if namespaces:
        acceptedNamespaces = set(namespaces.split(','))

    FORMAT = '%(levelname)s: %(message)s'
    logging.basicConfig(format=FORMAT)

    logger = logging.getLogger()
    if not quiet:
        logger.setLevel(logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)

    input_file = input

    if not Extractor.keepLinks:
        ignoreTag('a')

    # sharing cache of parser templates is too slow:
    # manager = Manager()
    # templateCache = manager.dict()

    if article:
        if templates:
            if os.path.exists(templates):
                with open(templates) as file:
                    load_templates(file)

        file = fileinput.FileInput(input_file, openhook=fileinput.hook_compressed)
        for page_data in pages_from(file):
            id, revid, title, ns, page = page_data
            Extractor(id, revid, title, page).extract(sys.stdout)
        file.close()
        return

    output_path = output
    if output_path != '-' and not os.path.isdir(output_path):
        try:
            os.makedirs(output_path)
        except:
            logging.error('Could not create: %s', output_path)
            return

    process_dump(input_file, templates, output_path, file_size,
                 compress, processes)


if __name__ == '__main__':

     seq,voc = get_wiki_dataset('nlwiki/20161220/nlwiki-20161220-pages-articles1.xml.bz2',25000)
     print len(seq), len(voc)
