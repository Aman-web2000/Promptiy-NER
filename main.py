from promptify import Prompter, OpenAI, Pipeline
from config import api_key, labels, nlp_model, domain,one_shot_labelled_training_data, description

model = OpenAI(api_key)
prompter = Prompter(nlp_model)
pipeline = Pipeline(prompter , model)

## Zero Shot Learning 
def ner_zero_shot(text):
    result= pipeline.fit(text_input=text,domain= domain, labels=None)

    return result[0]['text']

## Zero shot with Custom Tags
def _ner_zero_shot_tags(text):
    result = pipeline.fit(text_input= text, domain=domain, labels=labels)

    return result

## One Shot Learning
def ner_one_shot(text):
    result = pipeline.fit( text_input  = text,
                          domain      = domain,
                          examples    = one_shot_labelled_training_data,
                          labels      = labels)
    
    return result[0]['text']

## One Shot Learning
def ner_one_shot_desc(text):
    result = pipeline.fit( text_input  = text,
                          domain      = domain,
                          examples    = one_shot_labelled_training_data,
                          description = description,
                          labels      = labels)
    
    return result[0]['text']
