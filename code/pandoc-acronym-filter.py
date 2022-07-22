#!/usr/bin/python3

import os
from panflute import *
import re

acronyms = {}

refcounts = {}

def resolveAcronyms(elem, doc):
    if isinstance(elem, Span) and "acronym-label" in elem.attributes:
        label = elem.attributes["acronym-label"]
        
        if label in acronyms:
            # this is the case: "singular" in form and "long" in form:
            value = acronyms[label]
            
            form = elem.attributes["acronym-form"]
            if label in refcounts and "short" in form:
                if "singular" in form:
                    value = label
                else:
                    value = label + "s"
            
            elif "full" in form or "short" in form:
                # remember that label has been used
                if "short" in form:
                    refcounts[label] = True
                
                if "singular" in form:
                    value = value + " (" + label + ")"
                else:
                    value = value + "s (" + label + "s)"
            
            elif "abbrv" in form:
                if "singular" in form:
                    value = label
                else:
                    value = label + "s"
            
            return Span(Str(value))

def loadAcronyms():
    pattern = re.compile(r"\\newacronym(\[.*\])?\{(?P<label>[A-Za-z]+)\}\{([A-Za-z]+)\}\{(?P<value>[A-Za-z 0-9\-]+)\}")
    
    d = os.path.dirname(__file__)
    filename = os.path.join(d, './texmf/Acronyms.sty')
    with open(filename, 'r', encoding='utf-8') as acronymsFile:
        for line in acronymsFile:
            match = pattern.match(line)
            if match:
                acronyms[match.group('label')] = match.group('value')

def main(doc=None):
    loadAcronyms()
    return run_filter(resolveAcronyms, doc=doc)


if __name__ == "__main__":
    main()
