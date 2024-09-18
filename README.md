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

![Static Badge](https://img.shields.io/badge/Python-%237F52FF?style=for-the-badge&logo=Python&logoColor=white)
![Static Badge](https://img.shields.io/badge/PyTorch-%23FE7B7B?style=for-the-badge&logo=PyTorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/PyTorchXLA-%234DA651?style=for-the-badge&logo=PyG&logoColor=white)
![Static Badge](https://img.shields.io/badge/CometML-%234A2296?style=for-the-badge&logo=bitcomet&logoColor=white)
![Static Badge](https://img.shields.io/badge/TPU-%2325A162?style=for-the-badge&logo=turbo&logoColor=white)
![Static Badge](https://img.shields.io/badge/Git-%23EA330E?style=for-the-badge&logo=git&logoColor=white)

The language model was written from scratch in [PyTorch](https://pytorch.org/) and has a transformer architecture like modern language models.

The model was first trained to predict the most likely continuation of the text using the [fineweb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), and then finetuned on dialogues from the [dailydialogs dataset](https://huggingface.co/datasets/li2017dailydialog/daily_dialog) to be able to conduct a dialogue.

The model has more than 100 million parameters, so the training took place on the [Kaggle platform](https://www.kaggle.com/):
1. [Casual train](https://www.kaggle.com/code/danildolgov/casual-train)
2. [Finetune LLN](https://www.kaggle.com/code/danildolgov/llm-finetune)

Trained models are also on Kaggle. You can find them [here](https://www.kaggle.com/models/danildolgov/nanogpt/).

# Running
To try running the code, you need to download the project and run the python file <b>dialog_demo.py</b>. The models will download themselves if it is discovered that they are not in the project.

# Preview
Here is a demonstration of what a dialogue with a model will look like and what responses it gives.
![model preview](https://github.com/KoTeuKaSeeker/DialogLLM/blob/main/assets/dialogllm_preview.gif)

Below are examples of dialogues with the model. All of them are the result of the model's work, you can verify this by downloading the project and asking the same questions. <br>

Dialogue 1 demonstrates that even such a simple model has some understanding of the world: it was able to give its own definition of the concept of "computer", which even makes logical sense.
| Dialog 1 |
| ----------------------------------------------------------------------------------------- | 
|[SPEAKER]: What is a computer? Tell me<br> |
|[BOT]:  A computer is a record-keeping system . It keeps all the data in a single place . |
|[SPEAKER]: How can I use it?<br> |
|[BOT]: You simply put all your documents in one place and then it copies them into memory .|
|[SPEAKER]: Can I borrow one from you for a while?<br> |
|[BOT]: Yes , you can .<br> |
|[SPEAKER]: Thank you, I really appreciate it! Do you have a college degree?<br> |
|[BOT]: Yes , a bachelor's degree and a master's .<br> |

<br>

Dialogue 2 shows that the model is able to navigate simple everyday situations.

| Dialog 2 |
| ----------------------------------------------------------------------------------------- | 
|[SPEAKER]: Do you like apples? |
|[BOT]:  Yes , they're very popular in my neighborhood .  |
|[SPEAKER]: How much does one apple cost now? |
|[BOT]:  Six cents . That's a lot of money .  |
|[SPEAKER]: Yes, that's a lot. Shall we go to the store for them? |
|[BOT]:  Okay . Look , apples have a higher sales price in my neighborhood .  |
|[SPEAKER]: Then we can go get them somewhere else. |
|[BOT]:  Where else ?  |
|[SPEAKER]: I think they are cheaper in the city center. |
|[BOT]:  I see . |

<br>

Dialogue 3 is the most interesting - it demonstrates the model's understanding of the world. It is clear that the model has some understanding of the height of objects, but it does not understand at all what speed is.

| Dialog 3 |
| ----------------------------------------------------------------------------------------- | 
|[SPEAKER]: What is taller - a tree or a man? |
|[BOT]:  A tree.He is taller than me .  |
|[SPEAKER]: Who is lower - a man or a tree? |
|[BOT]:  A man and a half .  |
|[SPEAKER]: This is also true! And who is faster - a person or a car? |
|[BOT]:  Someone who can move fast , right ?  |
|[SPEAKER]: Right! |
|[BOT]:  Are you crazy ?  |
|[SPEAKER]: No, why are you asking? |
|[BOT]:  As a matter of fact , this man is taller than me .  |
|[SPEAKER]: Maybe. Could you walk faster than a car? |
|[BOT]:  Maybe . But I prefer a car myself .  |
|[SPEAKER]: Usually people drive cars because they move faster than people. |
|[BOT]:  Yeah , that ’ s right . But I drive a car because I want to be able to move fast . |
