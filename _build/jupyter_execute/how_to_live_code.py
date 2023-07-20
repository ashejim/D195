#!/usr/bin/env python
# coding: utf-8

# # Interactive Live Code Stuff
# 
# The html output can't driectly run stuff that needs Python to run -ti needs a kernel. To make your content interactive without requiring readers to leave the current page, you can use a project called Thebe. 
# 
# ## Using sphinx-thebe 
# 
# sphinx-thebe uses remote Jupyter kernels to execute your page’s code and return the results, and Binder to run the infrastructure for execution. You can do nearly anything with sphinx-thebe that you could do from within a Jupyter Notebook cell.

# <!-- Configure and load Thebe !-->
# <script type="text/x-thebe-config">
#   {
#       requestKernel: true,
#       mountActivateWidget: true,
#       mountStatusWidget: true,
#       data-executable: false,
#       binderOptions: {
#       repo: "binder-examples/requirements",
#       },
#   }
# </script>
# 
# <script type="text/javascript">
#     thebe.events.on("request-kernel")((kernel) => {
#         // Find any cells with an initialization tag and ask Thebe to run them when ready
#         kernel.requestExecute({code: "import numpy"})
#     });
# </script>
# 
# <script type="text/javascript">
#     thebe.events.on("request-kernel")(() => {
#         // Find any cells with an initialization tag and ask Thebe to run them when ready
#         var thebeInitCells = document.querySelectorAll('.thebe-init');
#         thebeInitCells.forEach((cell) => {
#             console.log("Initializing Thebe with cell: " + cell.id);
#             const initButton = cell.querySelector('.thebe-run-button');
#             initButton.click();
#         });
#     });
# </script>
#     
# <script src="https://unpkg.com/thebe@latest/lib/index.js"></script>
# 
# <div class="thebe-activate"></div>
# <div class="thebe-status"></div>

# In[1]:


print("false")


# Inserting html editable/runnanble code using Markdoen
# 
# <pre data-executable="true" data-output="true" data-language="python">print("Hello true!")</pre>

# Inserting code that's not runnable in html
# 
# <pre data-language="python">print("Hello false")</pre>
# 
# Inserting normal code that's not runnable unless a thebe-button is clicked.

# In[2]:


print('Hello, no thebe here 2')


# ```{code-block} python
# print("hello world!")
# ```

# ```{thebe-button}
# ```

# <div class="highlight">
#     <pre>print("hi!")</pre>
#  </div>

# <pre data-executable data-readonly>print("I cannot be modified")</pre>

# ## Interactive widgets using Python
# 

# In[3]:


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

def f(x):
    return x


# And some lines of stuff.... 
# and then the interact widget. It's necessary to run the imports through Thebe first. They don' load until Thebe is run and the then the code cell is run. 

# In[4]:


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

def f(x):
    return x

interact(f, x=10);


# <pre data-executable="true" data-readonly init = "true" thebe-init = "true">
# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets
# 
# def f(x):
#     return x
# 
# interact(f, x=10);
# print('hello 2')
# </pre>

# In[5]:


print("hi auto python")

##required
import pandas as pd
import numpy as np
##Have all necessary packages imported
#import some_packageA
#import some_packageB

##Run this line and send the output as .txt file. 
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))


# ```{code-block}
# :class: thebe, thebe-init
# print("hi MD")
# ```
