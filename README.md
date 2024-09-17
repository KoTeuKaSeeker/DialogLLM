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



# Running Locally

To run the project locally, you need to enter the commands below. To do this, you will need to open 3 instances of terminal that you will be using. 
Each instance will be associated with back-end, front-end or tensorflow/serving model.

> **Note**
> 
> In the following commands, the path to the newly cloned project will be specified as ``[project dir]``.
> 
> **Example**
> 
> Let's say you cloned this project into this directory:
> ```shell
> C:\Projects\
> ```
>  Then ``[project dir]`` is:
> ```shell
> C:\Projects\nlp-natural-disaster\
> ```
> So if you need to enter a command:
> ```shell
> cd [project dir]
> ```
> Then, instead, you should enter this:
> ```shell
> cd C:\Projects\nlp-natural-disaster\
> ```

### Terminal instance #1 (back-end)
Make sure that you have [Python](https://www.python.org/downloads/release/python-3115/) installed.
```shell
cd [project dir]
```
```shell
python -m venv venv
```
> **Important**
> 
> Windows Powershell:
> ```shell
> venv\Scripts\activate.ps1
> ```
>
> Windows Command Prompt:
> ```shell
> venv\Scripts\activate
> ```
```shell
pip install -r requirements.txt
```
```shell
cd back-end
```
```shell
python server.py -d
```

### Terminal instance #2 (front-end)
Make sure that you have [node.js](https://nodejs.org/en) installed.
```shell
cd [project dir]
```
```shell
cd front-end
```
```shell
npm install
```
```shell
npm start
```
### Terminal instance #3 (tensorflow/serving)
Make sure that you have [Docker](https://www.docker.com/) installed and running.
> **Note**
>
> ``[project dir]`` in this command should be substituted by you project directory path.
```shell
docker run --rm -it -v [project dir]\tensorflow-model\models:/models -p 8605:8605 --entrypoint /bin/bash tensorflow/serving
```
```shell
tensorflow_model_server --rest_api_port=8605 --model_name=nlp_natural_disaster --model_base_path=/models/
```
Your app should now be running on [localhost:4000](http://localhost:4000/).

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

# Preview
This is what the application looks like when it's running:
![app_preview](https://github.com/SpectreSpect/nlp-natural-disaster/assets/52841087/25cd1fc0-7c4a-4dcc-90c0-b3e69b12d2d7)
