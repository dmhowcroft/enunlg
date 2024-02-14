import stanza

nlp = stanza.Pipeline(lang="en", processors="tokenize,pos")

with open('enriched-webnlg.tokenized-texts', 'r') as in_file:
    docs = in_file.readlines()
docs = [doc.strip() for doc in docs]
out_docs = nlp.bulk_process(docs)
