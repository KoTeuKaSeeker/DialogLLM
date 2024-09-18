# Dialog LLM

[![Docker](https://img.shields.io/badge/Docker-%231D63ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![CSS3](https://img.shields.io/badge/CSS-%23214CE5?logo=css3&logoColor=white)](https://en.wikipedia.org/wiki/CSS)
[![HRML5](https://img.shields.io/badge/HTML5-%23E44D26?logo=HTML5&logoColor=white)](https://en.wikipedia.org/wiki/HTML5)
[![JavaScript](https://img.shields.io/badge/JavaScript-%23F7E018?logo=javascript&logoColor=white)](https://en.wikipedia.org/wiki/JavaScript)
[![ReactJS](https://img.shields.io/badge/ReactJS-%2311C8E8?logo=react&logoColor=white)](https://react.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF9300?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-%233572A5?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-grey?logo=flask&logoColor=white)](https://flask.palletsprojects.com/en/2.3.x/)
[![Git](https://img.shields.io/badge/Git-%23EA330E?logo=git&logoColor=white)](https://git-scm.com/)
[![TensorFlowServing](https://img.shields.io/badge/TensorFlow%2Fserving-%23F0910E?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/tfx/guide/serving)

The language model was written from scratch in [PyTorch](https://pytorch.org/) and has a transformer architecture like modern language models.

The model was first trained to predict the most likely continuation of the text using the [fineweb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), and then finetuned on dialogues from the [dailydialogs dataset](https://huggingface.co/datasets/li2017dailydialog/daily_dialog) to be able to conduct a dialogue.

The model has more than 100 million parameters, so the training took place on the [Kaggle platform](https://www.kaggle.com/):
1. [Casual train](https://www.kaggle.com/code/danildolgov/casual-train)
2. [Finetune LLN](https://www.kaggle.com/code/danildolgov/llm-finetune)

Trained models are also on Kaggle. You can find them [here](https://www.kaggle.com/models/danildolgov/nanogpt/).

# Running
To try running the code, you need to download the project and run the python file <b>dialog_demo.py</b>. The models will download themselves if it is discovered that they are not in the project. If this does not happen for some reason, you can download the model yourself from the link above and specify the path to the downloaded file.

# Preview
Here is a demonstration of what a dialogue with a model will look like and what responses it gives.
![model preview](https://github.com/SpectreSpect/nlp-natural-disaster/assets/52841087/25cd1fc0-7c4a-4dcc-90c0-b3e69b12d2d7)

# Techonlogies used

Here are all the technologies that were used to create this project:

> **Languages**
>
> [![JavaScript](https://img.shields.io/badge/JavaScript-%23F7E018?logo=javascript&logoColor=white)](https://en.wikipedia.org/wiki/JavaScript)
> [![Python](https://img.shields.io/badge/Python-%233572A5?logo=python&logoColor=white)](https://www.python.org/)
> [![HRML5](https://img.shields.io/badge/HTML5-%23E44D26?logo=HTML5&logoColor=white)](https://en.wikipedia.org/wiki/HTML5)
> [![CSS3](https://img.shields.io/badge/CSS-%23214CE5?logo=css3&logoColor=white)](https://en.wikipedia.org/wiki/CSS)


> **Frameworks/Libraries**
>
> [![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF9300?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
> [![Flask](https://img.shields.io/badge/Flask-grey?logo=flask&logoColor=white)](https://flask.palletsprojects.com/en/2.3.x/)
> [![ReactJS](https://img.shields.io/badge/ReactJS-%2311C8E8?logo=react&logoColor=white)](https://react.dev/)



> **Other**
>
> [![TensorFlowServing](https://img.shields.io/badge/TensorFlow%2Fserving-%23F0910E?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/tfx/guide/serving)
> [![Docker](https://img.shields.io/badge/Docker-%231D63ED?logo=docker&logoColor=white)](https://www.docker.com/)
> [![Git](https://img.shields.io/badge/Git-%23EA330E?logo=git&logoColor=white)](https://git-scm.com/)
