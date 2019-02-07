# NBA Social Power Estimator
Deep neural network to predict the influence of NBA players given a series of on-court statistics.
Output of this regression model is a prediction for the average number of retweets for a given player on their Twitter posts. Trained on the data of over 300 current NBA players.


Running this project:
1. Sort the raw data that has been compiled regarding NBA players and NBA teams.
```
python compile.py
```
2. Train the neural network:
```
python train.py
```


Potential uses of this project:
* Allows branding/endorsment companies to predict outreach of their clients.
* Allows players/managment teams, particularly for rookies or sophomores, to negotiate deals given an approximate figure for the reach they will attain.
