# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import openai
from api_secretes import API_KEY
MY_API_KEY = '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'
openai.api_key = MY_API_KEY
prompt = 'Give me the right option out of four to answer the question after reading the whole context. question: option: context: answer:'
openai.File.create(file = open('xx.json'), purpose ='classification')
openai.File.create(file = open('xx.json'), purpose ='answers')
response = openai.Completion.create(engine="text-davinci-001", prompt=prompt, max_token = 10)#max response token
#temperature : higher means the model take more risk
print(response)
'''
Authorization: Bearer MY_API_KEY

curl https://api.openai.com/v1/models \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'OpenAI-Organization: org-t6ygn1B2vF9oML9g9HCjCT5f'
'''
import os
import openai
openai.organization = "org-t6ygn1B2vF9oML9g9HCjCT5f"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()