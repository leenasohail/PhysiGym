# PhysiCell Python3 Embedding Reference Man Page

This is the technical descriptions of the machinery and how to operate it.


# py3pc_embedding: embedding PhysiCell into a Python3 module

References are maintained in each module's [docstring](https://en.wikipedia.org/wiki/Docstring).\
You can access them through the [source code](https://github.com/elmbeech/physicellembedding/blob/main/py3pc_embedding/custom_modules/physicellmodule.cpp#L449), or by first loading the physicell module.

```python3
from embedding import physicell
```
Then, for each physicell module, get on the fly reference information with the [help](https://en.wikipedia.org/wiki/Help!) command.

```python3
# all module functions
help(physicell)

# physicell control function
help(physicell.start)
help(physicell.step)
help(physicell.stop)

# interface function
help(physicell.get_parameter)
help(physicell.set_parameter)
help(physicell.get_variable)
help(physicell.set_variable)
help(physicell.get_vector)
help(physicell.set_vector)
help(physicell.get_cell)
help(physicell.get_microenv)

# operating system command line interface function
help(physicell.system)
```


# pcpy3_embedding: embedding the Python3 interpreter into the PhysiCell main loop

+ All Python3 code hast to be written into the [custom_modules/embedding.py](https://github.com/elmbeech/physicellembedding/blob/main/pcpy3_embedding/custom_modules/embedded.py) file.
+ The custom_modules/embedding.py file is called from the PhysiCell main loop within the [main.cpp](https://github.com/elmbeech/physicellembedding/blob/main/pcpy3_embedding/main.cpp#L221) main function.
