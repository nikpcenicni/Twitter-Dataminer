
# Twitter Data Miner and Sentiment Analyzer

This project was created as a way to determine the sentiment associated with a given tweet.
Users are able to enter username and either select tweet replies or mentions and analyze the sentiment that is associated with them.


## Deployment

Deploying this project has different steps depending on the technology used.

### Cloning the project
This project can be download directly from github.com by downloading as a zip, or grabing from the releases.

To Clone the repository using git cli run the following command.
```
gh repo clone nikpcenicni/Twitter-Dataminer
```

### Installation
Once the project has been cloned or downloaded it is ready to be installed.

Open repository in your terminal and follow the instructions below.

### PIP
If you are wishing to deploy this project with the default python follow these steps.


To create a virtual envrionment and install all required packages run the commands below in the cloned directory.
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Conda
If you are using Anaconda follow these instructions

```
conda create --name <env> --file conda-requirements.txt
```

You can replace ```<env>``` with a name of your choosing.


## Running

To run the project you must first activate the virtual environment.

### PIP
```
source env/bin/activate
```

### Conda
```
conda activate <env>
```
where ```<env>``` is the name of the environment you created.

Once the environment is activated you can run the project by running the following command.
```
python main.py
```

## Features

### Tweet Miner
The tweet miner is a tool that allows users to mine tweets from a given user. The user can select to mine tweets from the user's mentions or replies. The user can also select the number of tweets to mine.

### Sentiment Analyzer
The sentiment analyzer is a tool that allows users to analyze the sentiment of a given tweet. The user can select to analyze the sentiment of a tweet by entering the tweet id or by entering the tweet text. The user can also select the number of tweets to analyze.

## Built With
This project was built using the following technologies.

[Python](https://www.python.org/) - The programming language used

[Tweepy](https://www.tweepy.org/) - The python library used to interact with the Twitter API

[NLTK](https://www.nltk.org/) - The python library used to analyze the sentiment of the tweets

[Tensorflow](https://www.tensorflow.org/) - The python library used to train the sentiment analysis model

[SciKit Learn](https://scikit-learn.org/stable/) - The python library used to train the sentiment analysis model

[Matplotlib](https://matplotlib.org/) - The python library used to create the graphs

[Pandas](https://pandas.pydata.org/) - The python library used to create the graphs

[Seaborn](https://seaborn.pydata.org/) - The python library used to create the graphs

[Transformers](https://huggingface.co/transformers/) - The python library used to train the sentiment analysis model

[imbalance-learn](https://imbalanced-learn.org/stable/) - The python library used to train the sentiment analysis model

[Jaal](https://github.com/imohitmayank/jaal) - Python library for viewing user network graphs

[Dotenv](https://pypi.org/project/python-dotenv/) - Python library for loading environment variables from .env files


## Authors
Nikholas Pcenicni - Initial work - [nikpcenicni](https://pcenicni.dev)

Arshjit Pelia - Initial work - [Arxh-23](https://github.com/Arxh-23)
