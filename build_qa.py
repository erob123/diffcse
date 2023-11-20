# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')
from tqdm import tqdm
import re

import faiss
# encoded_data = model.encode(df.Plot.tolist())
# encoded_data = np.asarray(encoded_data.astype('float32'))
# index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
# index.add_with_ids(encoded_data, np.array(range(0, len(df))))
# faiss.write_index(index, 'movie_plot.index')

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

torch.set_default_device('cuda') 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





tokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')


# para = "Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects."

# input_ids = tokenizer.encode(para, return_tensors='pt')
# with torch.no_grad():
#     outputs = model.generate(
#         input_ids=input_ids,
#         max_length=64,
#         do_sample=True,
#         top_p=0.95,
#         num_return_sequences=3)

# print("Paragraph:")
# print(para)

# print("\nGenerated Queries:")
# for i in range(len(outputs)):
#     query = tokenizer.decode(outputs[i], skip_special_tokens=True)
#     print(f'{i + 1}: {query}')









with open("cui/(CUI) MM_chats_20230620_message_only.csv","r") as file:
    sourceFileContents = file.read()

while '\n\n\n' in sourceFileContents:
    sourceFileContents = re.sub(r'\n\s*\n', '\n\n', sourceFileContents)


count = 0
with open("cui/(CUI) MM_chats_20230620_message_only.csv","r") as file:
    lines = file.readlines()

paragraphs = []
new_para = ""
for i, line in enumerate(lines):

    if line != '\n':
        new_para += " " + line

    if i % 8 == 0 and i > 0:
        paragraphs.append(new_para)
        new_para = ""

# Parameters for generation
batch_size = 16 #Batch size
num_queries = 5 #Number of queries to generate for every paragraph
max_length_paragraph = 512 #Max length for paragraph
max_length_query = 64   #Max length for output query
def _removeNonAscii(s): return "".join(i for i in s if ord(i) < 128)
with open('generated_queries_all.tsv', 'w') as fOut:
    for start_idx in tqdm(range(0, len(paragraphs), batch_size)):
        end_index = min([start_idx+batch_size, len(paragraphs)])
        sub_paragraphs = paragraphs[start_idx:end_index]
        inputs = tokenizer.prepare_seq2seq_batch(sub_paragraphs, max_length=max_length_paragraph, truncation=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length_query,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=num_queries)

        for idx, out in enumerate(outputs):
            query = tokenizer.decode(out, skip_special_tokens=True)
            query = _removeNonAscii(query)
            para = sub_paragraphs[int(idx/num_queries)]
            para = _removeNonAscii(para)
            fOut.write("{}\t{}\n".format(query.replace("\t", " ").strip(), para.replace("\t", " ").strip()))