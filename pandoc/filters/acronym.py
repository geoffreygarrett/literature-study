#!/usr/bin/env python3


"""
Panflute filter that allows for acronyms in latex
Usage:
- In markdown, use it as links: [SO](acro "Stack Overflow")
- When outputting to latex, you must add this line to the preamble:
\\usepackage[acronym,smallcaps]{glossaries}
- Then, this filter will add "\newacronym{LRU}{LRU}{Least Recently Used} for the definition of LRU and finally \gls{LRU} to every time the term is used in the text."
(see https://groups.google.com/forum/#!topic/pandoc-discuss/Bz1cG55BKjM)
"""

from string import Template  # using .format() is hard because of {} in tex
import panflute as pf

TEMPLATE_GLS = Template(r"\gls{$acronym}")
TEMPLATE_NEWACRONYM = Template(r"\newacronym{$acronym}{$acronym}{$definition}")

import re 
import os 
import sys 
def prepare(doc):
    """
    In order to deal with acronyms, we need to load and parse the acronyms.tex manually.
    """
    doc.acronyms = {}
    doc.refcounts = {}
    pattern = re.compile(r"\\newacronym(\[.*\])?\{(?P<label>[A-Za-z-]+)\}\{.+\}\{(?P<value>[A-Za-z 0-9\-]+)\}")
    # write to std out so that we can parse it manually
    
    with open('./glossaries/general.tex', 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                doc.acronyms[match.group('label')] = match.group('value')
    

def action(elem, doc):
    if isinstance(elem, pf.Span) and "acronym-label" in elem.attributes:
        label = elem.attributes["acronym-label"]
        
        if label in doc.acronyms:
            # this is the case: "singular" in form and "long" in form:
            value = doc.acronyms[label]
            
            form = elem.attributes["acronym-form"]
            if label in doc.refcounts and "short" in form:
                if "singular" in form:
                    value = label
                else:
                    value = label + "s"
            
            elif "full" in form or "short" in form:
                # remember that label has been used
                if "short" in form:
                    doc.refcounts[label] = True
                
                if "singular" in form:
                    value = value + " (" + label + ")"
                else:
                    value = value + "s (" + label + "s)"
            
            elif "abbrv" in form:
                if "singular" in form:
                    value = label
                else:
                    value = label + "s"
            
            return pf.Span(pf.Str(value))


import os 
def finalize(doc):
    if doc.format == 'latex':
        tex = [r'\usepackage[acronym,smallcaps]{glossaries}', '\makeglossaries']
        for acronym, definition in doc.acronyms.items():
            tex_acronym = TEMPLATE_NEWACRONYM.safe_substitute(acronym=acronym, definition=definition)
            tex.append(tex_acronym)

        tex = [pf.MetaInlines(pf.RawInline(line, format='latex')) for line in tex]
        tex = pf.MetaList(*tex)
        if 'header-includes' in doc.metadata:
            doc.metadata['header-includes'].content.extend(tex)
        else:
            doc.metadata['header-includes'] = tex

    # if doc.format == 'gfm':
    #     for acronym, definition in doc.acronyms.items():
            # os.std.err.write('{} {}\n'.format(acronym, definition))


def main(doc=None):
    return pf.run_filter(action, prepare=prepare, finalize=finalize, doc=doc) 


if __name__ == '__main__':
    main()