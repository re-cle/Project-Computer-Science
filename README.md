# Python Project for the Module: Project - Computer Science
Contains all code referenced during the report, including the used models.
#### Code Snippets from the report are in the 'CodeSnippets' folder, figures used are in the 'Figures' folder
#### The 'Tweets.csv' file contains the Tweets used to train the model and generate predictions

## Setup Guide of the Project for Python3.12
### 1. Move to path where the project shall be cloned and execute:
  

    git clone https://github.com/re-cle/Project-Computer-Science.git

### 2. Install pip for Python3.12:
		

##### Note: might vary slightly depending on the CLI
		
    python -m ensurepip --upgrade


### 3. Install required modules: 

 ##### If in main folder of project:
 
    python3 -m pip install -r requirements.txt
  
 ##### or, if not in project folder:
	
    python3 -m pip install -r <pathToProject>/requirements.txt


### 4. To run the machine learning model:
  ##### If in project folder:
    python3 MachineLearningModel.py
  ##### If not in project folder:
    python3 <pathToProject>/MachineLearningModel.py

### 5. To run the deep learning model:
  ##### If in project folder:
    python3 DeepLearningModel.py
  ##### If not in project folder:
    python3 <pathToProject>/DeepLearningModel.py
