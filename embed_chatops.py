from diffcse import DiffCSE
import pandas as pd

# model = DiffCSE("voidism/diffcse-bert-base-uncased-sts", pooler="cls_before_pooler")
model = DiffCSE("cui/models/alexa", pooler="cls_before_pooler")
# embeddings = model.encode("A woman is reading.")

# sentences = ['What is the status of RCH535?', 'What is the status of RCH715?']
# model.build_index(sentences)
# results = model.search("Has RCH535 taken off?")
# print(f"results: {results}")

# df = pd.read_csv("cui/(CUI) MM_chats_20230620_channel_and_message.csv")
# df['message'][3]

# df['sentences'] = df['channelname'].astype(str) + ',' + df['message']
# print(df['sentences'])

# sentences = df['message'].to_numpy(dtype=str).tolist()
# model.build_index(sentences)
# results = model.search("- RCH243 - Has RCH243 taken off?")
# results

# import numpy as np
# np.genfromtxt("cui/full_text.txt")


count = 0
with open("cui/full_text.txt","r") as file:
    lines = file.readlines()

model.build_index(lines)
results = model.search("class B airspace")
results

# df['channelname'].append(df['message']).to_csv("out.csv",index=False,header=False)

# trained longer but loss really hits a floor around 2 epochs