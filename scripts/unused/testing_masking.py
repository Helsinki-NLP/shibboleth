from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

sentence = "Banja Luka sevap ekmek raja"

labels = 4

tokenizer = AutoTokenizer.from_pretrained('classla/bcms-bertic')
classifier = AutoModelForSequenceClassification.from_pretrained('../explainability_bcms/model_twitter_eqcl_bertic/best', num_labels=labels)
        
repl = '[MASK]'

# Find mask token ID
batch_encoding = tokenizer(repl)
if len(batch_encoding['input_ids']) == 3:
    mask_id = batch_encoding['input_ids'][1]
        
# Tokenize the sentence
#tokenized_sent = tokenizer(sentence) # returns a dict; key of interest: token_ids
input_ids = tokenizer(sentence)["input_ids"]
print(input_ids)
tens = torch.tensor([input_ids])
print(tens)
    
with torch.no_grad():
    #logits = classifier(**tokenized_sent).logits
    logits = classifier.forward(input_ids=tens).logits
    predicted_class_id = logits.argmax().item()
    print(predicted_class_id)
    #model.config.id2label[predicted_class_id]


    
#outputs = classifier.eval(tokenized_sent['input_ids'])
#print(outputs)
        
# encoded_sequence = tokenizer(sequence)["input_ids"]
# forward(input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None)
# The BertForSequenceClassification forward method, overrides the __call__() special method.
    
    