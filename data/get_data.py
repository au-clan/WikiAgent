import requests, sys

def main ():

    link = 'https://snap.stanford.edu/data/wikispeedia/wikispeedia_paths-and-graph.tar.gz'
    file_name = 'data/wikispeedia_paths-and-graph.tar.gz'
    with open(file_name, "wb") as f:
        print("Downloading %s" % file_name)
        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None: # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                sys.stdout.flush()

    link = 'https://snap.stanford.edu/data/wikispeedia/wikispeedia_articles_html.tar.gz'
    file_name = 'data/wikispeedia_articles_html.tar.gz'
    with open(file_name, "wb") as f:
        print("\nDownloading %s" % file_name)
        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None: # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                sys.stdout.flush()

if __name__ == '__main__':
    main()