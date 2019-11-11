# *chloe*

<img src="https://raw.githubusercontent.com/dwyl/repo-badges/master/highresPNGs/start-with-why-HiRes.png" height="20" width="100">

Our philosophies:

"The most beautiful experience we have is the mysterious. It is the  emotion that stands at the cradle of all art and science" -Albert Einstein

"Anyone can cook" -that French chef from Ratatouille

"Less bragging and flash, more science and math " -*chloe* 

"Famous quotes are pretentious" -Vicki 

Most AI research today is done by elite universities and corporate labs. The pursuit of science and creativity should be a viable project for any person, from any walk of life, who is excited by that feeling of mystery and building something that grows  

To make new research approachable, silly and fun to the motivated beginner programmer  with at least a high school calculus level of math, we will be building chloe from the basics, from linear algebra, probabilty theory, etc, all the way to the latest AI research, at each step putting our learning into *chloe* so we remember it forever. *chloe* is an end to end conversational neural network or chatbot. She is built from modules that are thoroughly explained and demonstrated using jupyter notebooks and toy examples that build from these basics. 

[Support *chloe* on Patreon](https://www.patreon.com/chloerobotics)

[Python tutorials](https://www.learnpython.org/) are everywhere on the internet. If you want to learn so you can see how Deep Learning Transformers work for NLP I suggest using [jupyter notebooks](https://youtu.be/pxPzuyCOoMI) like [this one](https://www.dataquest.io/blog/jupyter-notebook-tutorial/). For a visual intro to linear algebra I recommend [3Blue1Brown's Essence of linear algebra](https://youtu.be/fNk_zzaMoSs) and [neural networks](https://youtu.be/aircAruvnKk)

## Table of Contents

- [START_HERE](START_HERE.ipynb) introduction, summon and meet *chloe*
- [TALK](notebooks/Talk.ipynb) tongue-tied, joint probability distribution and beam search **new under construction**
- More on the way

### How to Start

I am serious when I say "build advanced concepts from the very basics", so bare with me. **more detailed setup instructions under construction**

install [virtual environment](https://towardsdatascience.com/virtual-environments-104c62d48c54) then create a python 3.6 virtual environment

`$ python3 -m venv env`

Whenever you want to modify the code, activate virtual environment with python 3.6 inside the same folder as your environment env using 

`$ source env/bin/activate`

install dependencies

`$ pip install -r requirements.txt`

- Python 3.6
- torch==1.3.0 (PyTorch)
- nltk==3.4.5 (Natural Language Toolkit)

save new dependences to requirements

`$ pip freeze > requirements.txt`

You can deactivate the virtual environment using the following command in your terminal:

`$ deactivate`
