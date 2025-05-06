# Python AI

Create AI Brain with PythonAI

## Install

```python
pip install PythonAI
```
 Or,

```python
pip install pyai
```

## Import Brain From pyai module

```python
from pyai import Brain
```

## How to use it

```python
from pyai import Brain

brain = Brain('intents.json') # Enter Path of your dataset.

while True:
	message = input('Message : ')
	message_type = brain.predict_message_type(message)
	
	if message_type == 'Question':
		pyai_say(process_messages(message))
```

And So on...
