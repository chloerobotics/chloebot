# *chloe*

<img src="https://raw.githubusercontent.com/dwyl/repo-badges/master/highresPNGs/start-with-why-HiRes.png" height="20" width="100">

## What is this?

Most AI research today is done by elite universities and corporate labs. The pursuit of science and creativity should be a viable project for any person, from any walk of life, who is excited by that feeling of mystery and building something that grows  

To make new research approachable, silly and fun to the motivated beginner programmer  with at least a high school calculus level of math, we will be building chloe from the basics, from linear algebra, probabilty theory, etc, all the way to the latest AI research, at each step putting our learning into *chloe* so we remember it forever. *chloe* is an end to end conversational neural network or chatbot. She is built from modules that are thoroughly explained and demonstrated using jupyter notebooks and toy examples that build from these basics. 

## Our philosophies

"The most beautiful experience we have is the mysterious. It is the  emotion that stands at the cradle of all art and science" -Albert Einstein

"Anyone can cook" -that French chef from Ratatouille

"Less bragging and flash, more science and math " -*chloe* 

"Famous quotes are pretentious" -Vicki 

## What if I'm a Noob?

Youre a noob? thats awesome! 

[Python tutorials](https://www.learnpython.org/) are everywhere on the internet. If you want to learn so you can see how Deep Learning Transformers work for NLP I suggest using [jupyter notebooks](https://youtu.be/pxPzuyCOoMI) like [this one](https://www.dataquest.io/blog/jupyter-notebook-tutorial/). For a visual intro to linear algebra I recommend [3Blue1Brown's Essence of linear algebra](https://youtu.be/fNk_zzaMoSs) and [neural networks](https://youtu.be/aircAruvnKk)

## NLP Stack 

- Python 3.6
- torch==1.3.0 (PyTorch)
- torchtext 
- nltk==3.4.5 (Natural Language Toolkit)

## Table of Contents

- [START_HERE](START_HERE.ipynb) introduction, summon and meet *chloe*
- [TALK](notebooks/Talk.ipynb) tongue-tied, joint probability distribution and beam search **new under construction**
- More on the way

## How to Start

I am serious when I say "build advanced concepts from the very basics", so bare with me. A more detailed setup instructions is on the way. 


if you already have python 3.6 and virtual environments, create a python 3.6 virtual environment, here i used env36 for python3.6 but you can use anything

`$ python3.6 -m venv env36`

otherwise it is simple to get python3.6 and virtual environments

[how to install Python 3.6 on ubuntu](http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/)

install [virtual environment](https://towardsdatascience.com/virtual-environments-104c62d48c54) then 

[how to specify the Python executable you want to use](https://stackoverflow.com/questions/1534210/use-different-python-version-with-virtualenv)


`sudo add-apt-repository ppa:jonathonf/python-3.6`

`sudo apt-get update`

`sudo apt-get install python3.6`

`virtualenv --python=/usr/bin/python3.6 env36`


Whenever you want to modify the code, activate virtual environment with python 3.6 inside the same folder as your environment env using 

`$ source env36/bin/activate`

install dependencies

`$ pip install -r requirements.txt`

[even with virtual environments, some troubleshoot might be needed](https://github.com/tensorflow/tensorflow/issues/559)

[with enough google searches you can find an answer for almost any problem](https://stackoverflow.com/questions/45912674/attributeerror-module-numpy-core-multiarray-has-no-attribute-einsum)

save new dependences to requirements

`$ pip freeze > requirements.txt`

You can deactivate the virtual environment using the following command in your terminal:

`$ deactivate`

## More Tips and Tricks

if you get a `ImportError: No module named` while at the saem time in your Terminal you get 

`pip3 install import-ipynb`

`Requirement already satisfied: import-ipynb in /media/chloe/New Volume/Chloe/chloebot/env36/lib/python3.6/site-packages (0.1.3)`

This can be fixed by providing your python interpreter with the path-to-your-module,the path 

`import sys`

`sys.path.append('/media/carson/New Volume/Chloe/chloebot/env36/lib/python3.6/site-packages')` 

## How can I help you or get help from you?

[Support *ChloeRobotics* on Patreon and send us a message](https://www.patreon.com/chloerobotics)