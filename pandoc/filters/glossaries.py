# this script will read all .tex files found in <directory> and
# regex search for the following patterns:
# \newglossaryentry{<key>}{<short>}{<long>}
# \newacronym{<key>}{
# 	type=<type>,
# 	name=<name>,
# 	sort=<sort>,
# 	description=<description>
# }

DATA_REGEX = [
    r'\\newacronym\{([\n\s]+)?(?P<key>.*?)([\n\s]+)?\}\{([\n\s]+)?(?P<short>.*?)([\n\s]+)?\}\{([\n\s]+)?(?P<long>.*?)([\n\s]+)?\}',
    r'\\newglossaryentry\{(?P<key>.*?)\}\{\n+\s+type=(?P<type>.*?),\n+?\s+?name=(?P<name>.*?),\n+?\s+?sort=(?P<sort>.*?),\n+?\s+?(description=\{(?P<description>.*?))?\},?\}',
    ]

def main():

    import os
    import re
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Read all .tex files in <directory> and regex search for the following patterns: \newglossaryentry{<key>}{<short>}{<long>} \newacronym{<key>}{\n\ttype=<type>,\n\tname=<name>,\n\tsort=<sort>,\n\tdescription=<description>\n}')
    parser.add_argument('directory', help='directory to search for .tex files')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()

    if args.verbose:
        print('Searching for .tex files in ' + args.directory)

    # get all .tex files in directory
    tex_files = []
    for file in os.listdir(args.directory):
        if file.endswith(".tex"):
            tex_files.append(file)

    # search for patterns in each .tex file
    for file in tex_files:
        if args.verbose:
            print('Searching ' + file)
        with open(args.directory + '/' + file, 'r') as f:
            
            # search for patterns in file
            for regex in DATA_REGEX:
                for match in re.finditer(regex, f.read(), re.MULTILINE):
                    print(match.groupdict())

            # if match:
            #     print('\t' + match.group(1) + '\t' + match.group(2) + '\t' + match.group(3))
            # # search for \newacronym
            # match = re.search(r'\\newacronym\{(.*?)\}\{(.*?)\}', line)
            # if match:
            #     print('\t' + match.group(1) + '\t' + match.group(2))
            

if __name__ == "__main__":
    main()