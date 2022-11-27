
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


