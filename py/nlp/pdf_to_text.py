import re
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import os


def test(pdfpath, txtpath, buf=True):
    rsrcmgr = PDFResourceManager()
    outfp = StringIO()
    laparams = LAParams()
    laparams.detect_vertical = True
    device = TextConverter(rsrcmgr, outfp, codec='utf-8', laparams=laparams)
    fp = open(pdfpath, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.get_pages(fp, pagenos=None, maxpages=0, caching=True, check_extractable=True):# maxpages：ページ指定（0は全ページ）
        interpreter.process_page(page)
    text = re.sub(re.compile(r"[ 　]+"), "", outfp.getvalue())
    fp.close()
    device.close()   
    outfp.close()
    print(text)
    f=open(txtpath,'w', encoding='utf-8')
    f.write(text)
    f.close()

proposal_files = os.listdir('/-/-/')
estimate_files = os.listdir('/-/-/')


for file in estimate_files:
    test(pdfpath="/-/-/", txtpath= "/-/-/")
