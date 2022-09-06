# command line tool which extracts all text from a pdf file and writes it to a text file.

import argparse
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    # codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text

import re

if __name__ == "__main__":

    # setup argparse for pdf target
    parser = argparse.ArgumentParser(description='Extract text from pdf file.')
    parser.add_argument('input', help='input pdf file')
    # output file is optional, if not specified, output to stdout
    parser.add_argument('output', help='output text file', nargs='?', default=None)

    args = parser.parse_args()

    # if no out is specified, take name from pdf
    if args.output is None:
        args.output = args.input.replace('.pdf', '.txt')

    # convert pdf to text
    text = convert_pdf_to_txt(args.input)

    # for 
    # - case_1: two consecutive newlines, ignore, 
    # - case_2: for one, replace with space,
    # - case_3: for one breaking a hyphen, remove hyphen and space
    
    # remove hypenate and newline
    text = re.sub(r'(?<=[a-z])-\n+(?=[a-z])', r'', text)

    # remove newlines after sentences
    text = re.sub(r'(.*)\n(?!\n)', r'\1 ', text)
    
    # remove fi
    text = re.sub(r'ﬁ', r'fi', text)
    text = re.sub(r'\u0005\s?', r'fi', text)


    # # remove lines that start with a space
    # # text = re.sub(r'\u000c', r'', text)
    # text = re.sub(r'(?<=\n)([\u000c\s]+)', r'\n', text)

    # # remove figures
    # text = re.sub(r'FIGURE.*', r'\n', text)

    # # add newline before and after title
    # text = re.sub(r'(?<=\n)\d+\s(?=[a-z])', r'', text)
    # text = re.sub(r'(?<=[^\.])\n+[-\d\.A-Z\s\u0006\:]+\n+(?=[a-z])', r' ', text)
    
    # # replace times unicode character
    # text = re.sub(r'[\u0002]+', r'${\\times}$', text)

    # # replace plus minus unicode character
    # text = re.sub(r'[\u0006]+', r'${\\pm}$', text)

    # # replace minus unicode character
    # text = re.sub(r'[\u0000]+', r'-', text)

    # # replace lonely page numbers
    # text = re.sub(r'(?<=\n)(\d+)(?=\n)', r'\n', text)

    # # place equal sign in math mode
    # text = re.sub(r'¼', r'=', text)

    # # replace textemdash
    # text = re.sub(r'(?<=[\d\%])(e)(?=[\d\%])', r'{\\textemdash}', text)
    
    # # newlines before and after title    
    # text = re.sub(r'\n*(\d+\.\d+?\s[A-Z].*)\n*', r'\n\n\1\n\n', text)

    # # replace errorenous dash in captilaized words
    # text = re.sub(r'(?<=[A-Z])(e)(?=[A-Z])', r'-', text)

    # text = re.sub(r'\n+1. A BRIEF HISTORY OF SPACECRAFT MISSIONS\n+', r'\n', text)
    # text = re.sub(r'(?<=[a-z])\n+(?=[a-z])', r' ', text)

    # # text = re.sub(r'(?<=[a-z])\n+(?=[a-z])', r' ', text)

    # # remove hypenate and newline
    # text = re.sub(r'(?<=[a-z])-\n+(?=[a-z])', r'', text)

    

    
    # text = re.sub(r'(?<=[^\.])\n+[\d\.A-Z\s^\x00-\x7F]+\n+(?=[a-z])', r' ', text)

    # text = re.sub(r'[^\.]\n(.*)\n\w.*\n+', r'\1', text)
    # text = re.sub(r'-\s*?\n\s*?(?!\n)', r'', text)
    # text = re.sub(r'\s*?\n\s*?(?!\n)', r' ', text)

    # write text to output file
    with open(args.output, 'w') as f:
        f.write(text)

    print('Wrote text to ' + args.output)

