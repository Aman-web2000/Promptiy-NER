from promptify import Prompter, OpenAI
from config import api_key, labels, nlp_model, domain,one_shot_labelled_training_data, description

model = OpenAI(api_key)
nlp_prompter = Prompter(model)

## Zero Shot Learning 
def ner_zero_shot(text):
    result= nlp_prompter.fit(nlp_model, domain= domain, text_imput=text, labels=labels)
    
    return eval(result['text'])

## One Shot Learning
def ner_one_shot(text):
    result = nlp_prompter.fit(nlp_model,
                          domain      = domain,
                          text_input  = text,
                          examples    = one_shot_labelled_training_data,
                          description = description,
                          labels      = labels)
    
    return eval(result['text'])