

# regex and replace all escaped underscores (\_) in url strings with unescaped (_)
import re

re_url = re.compile(r'((http|https)\:\/\/)([a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\\\.\&\/\?\:@\-_=#])*)')


file = "../library.bib"

if __name__ == "__main__":
    # read file
    with open(file, 'r') as f:
        lines = f.read()

    # find all urls 
    urls = re.findall(re_url, lines)

    for url in urls:
        s = url[2]
        if "\\_" in s:
            print(s)    
            r = s.replace(r'\_', r'_')
            print(r)
            lines = lines.replace(s, r)

    # write file
    with open("out2.bib", 'w') as f:
        f.writelines(lines)
    
    # print file

