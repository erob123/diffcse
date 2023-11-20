import numpy as np
import pandas as pd
import time

import faiss
encoded_data = model.encode(paragraphs)
encoded_data = np.asarray(encoded_data.astype('float32'))
index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
index.add_with_ids(encoded_data, np.array(range(0, len(paragraphs))))
faiss.write_index(index, 'chat.index')

    
def search(query, top_k, index, model):
    t=time.time()
    query_vector = model.encode([query])
    top_k = index.search(query_vector, top_k)
    print('>>>> Results in Total Time: {}'.format(time.time()-t))
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    results =  [paragraphs[idx] for idx in top_k_ids]
    return results


from pprint import pprint
query="what is the status of RCH715?"
results=search(query, top_k=200, index=index, model=model)
print("\n")
for result in results:
    print('\t',result)