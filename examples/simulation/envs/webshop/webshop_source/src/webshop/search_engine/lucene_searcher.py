import json
from pyserini.search.lucene import LuceneSearcher
from rich import print


searcher = LuceneSearcher('indexes')
hits = searcher.search('rubber sole shoes', k=20)

for hit in hits:
    doc = searcher.doc(hit.docid)
    print(doc)
    obj = json.loads(doc.raw())['product']['Title']
    print(obj)

print(len(hits))
