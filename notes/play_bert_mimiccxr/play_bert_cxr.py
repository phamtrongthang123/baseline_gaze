import torch
from transformers import AutoModel, AutoTokenizer
from torch import nn 
# import optimizer 
# import torch.nn.functional as F
import torch.optim as optim


# Load the model and tokenizer
url = "microsoft/BiomedVLP-CXR-BERT-specialized"
tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
model = AutoModel.from_pretrained(url, trust_remote_code=True)
fc = nn.Linear(768, 30522)
# Input text prompts (e.g., reference, synonym, contradiction)
# để autoregressive thì 1 là gán [mask] lúc end, 2 là thử 1 lớp decoder, nhớ freeze cái pretrained bert.
text_prompts_input = ["There is no pneumothorax or pleural [MASK]",
                "No pleural effusion or pneumothorax is [MASK]",
                "The extent of the pleural effusion is [MASK]"]
text_prompts = ["There is no pneumothorax or pleural ij [UNK]",
                "No pleural effusion or pneumothorax is picc",
                "The extent of the pleural effusion is constant."]
# Tokenize and compute the sentence embeddings
tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=text_prompts_input,
                                               add_special_tokens=True,
                                               padding='longest',
                                               return_tensors='pt')
adam = optim.Adam(fc.parameters(), lr=0.001)
while True:
    adam.zero_grad()
    token_inp, token_out = tokenizer_output.input_ids[:, :-1], tokenizer_output.input_ids[:, 1:]
    mask_inp, mask_out = tokenizer_output.attention_mask[:, :-1], tokenizer_output.attention_mask[:, 1:]
    embeddings = model(input_ids=token_inp,attention_mask=mask_inp, output_cls_projected_embedding=False, return_dict=True)
    logit = fc(embeddings.last_hidden_state)
    loss = nn.CrossEntropyLoss()(logit.reshape(-1, logit.size(-1)), token_out.reshape(-1))
    loss.backward()
    adam.step()
    print(loss.item())

print(tokenizer_output)
print(embeddings)
# Compute the cosine similarity of sentence embeddings obtained from input text prompts.
sim = torch.mm(embeddings, embeddings.t())
