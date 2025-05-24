# Python AI

Create AI Brain with PythonAI.
Easy to use and opensource. 

## Install

```python
pip install pythonaibrain
```
Or,

```python
git clone https://github.com/World-Of-Programming-And-Technology/PythonAI.git
```

```python
cd PythonAI
```

Install requirements. 

```python
pip install -r requirements.txt
```

## Import Brain From pyai module

```python
from pyai import Brain
```

## How to use it

```python
from os import system
from pyai import Brain

brain = Brain('intents.json') # Enter Path of your dataset.

while True:
	message = input('Message : ')
	message_type = brain.predict_message_type(message)
	
	if message_type == 'Question' or message_type == "Answer":
		brain.pyai_say(brain.process_messages(message))

	if message_type == "Shutdown":
		system(message)
```

And So on...

## Why Do We Use It.

It can help you to perform AI projects based on NLP (Nature Language Processing), ML (Mechine Learning) and DL (Deep Learning).

> You can also make your own AI Assistance with it.
> Give Data in intents.json file.

---

### Way to create intents.json file.

```json

{
  "intents" :[
        {
        "tag" : "gretting",
        "patterns" : ["Hi", "Hello", "Hey"],
        "responses": ["Hi, How can I assist you today", "Hey there"]
         },
         {
        "tag" : "gretting",
        "patternss" : ["Hi", "Hello", "Hey"],
        "response": ["Hi, How can I assist you today", "Hey there"]
          },
       ]
}
```

In this way you can create your intents.json  file. If you want you can add more data also.

> Add in this formate only, otherwise program will through an error. (Remember)

## Python AI functions and objects. 

| Function And Object Name| Description |
| :---: | :---: |
| Brain | It is an object to create *Barin* by passing *intents.json* file |
| AdvanceBrain | It also create *Barin* by passing *intents.json* file, but it is advance then ***Brain*** object. |
| pyai_say | It is a function inside Brain class to print the answer of the given question. |
| predict_message_type | It is use to predict the type  of questions Like the message in *question format* or *answer* or *command* or *start* or *shutdown*. |
| predict_entities | It stract *name*, *age* and *location* form the given message. |

---

## Import AdvanceBrain from pyai

You can also use AdvanceBrain in it. It is Advance then Brain class in pyai

## How to use AdvanceBrain.

```python

from pyai import AdvanceBrain

advance_brain = AdvanceBrain('intents.json') # You can also do AdvanceBrain() it also work.

while True:
	message = input('Message : ')
	message_type = advance_brain.predict_message_type(message) # It predict the message type.

	if message_type == "Question" or message_type == "Answer":
		advance_brain.pyai_say(advance_brain.process_messages(message)) # It can process the input message and print the output.

```

And so no...

## How to create ***_intents.json_*** file.

The same process

```json

{
  "intents" :[
        {
        "tag" : "gretting",
        "patterns" : ["Hi", "Hello", "Hey"],
        "responses": ["Hi, How can I assist you today", "Hey there"]
         },
         {
        "tag" : "gretting",
        "patternss" : ["Hi", "Hello", "Hey"],
        "response": ["Hi, How can I assist you today", "Hey there"]
          },
       ]
}
```

> Add in this formate only, otherwise program will through an error. (Remember)
