#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 08:53:51 2019

@author: jaisi8631
"""

# imports
import numpy as np
import pandas as pd


# remove unneccessary indexing columns
data = pd.read_csv('data/nba_2017_players_stats_combined.csv')
data = data.drop('Index', 1)
data = data.drop('Rk', 1)


# add column indicating salary
salary = pd.read_csv('data/nba_2017_salary.csv')
salaries = []
for i in range(0, data.shape[0]):
    player = data['PLAYER'][i]
    index = -1
    j = 0
    while(index == -1 and j < salary.shape[0]):
        if(salary['NAME'][j] == player):
            index = j
            salaries.append(salary['SALARY'][j])
        j += 1
    if(index == -1):
        salaries.append(-1)
        
        
# use the average salary for players whose salary cannot be found
nonNegative = [i for i in salaries if i >= 0]
avg = np.mean(nonNegative)
for i in range(0, len(salaries)):
    if(salaries[i] == -1):
        salaries[i] = avg


# create dataframe indicating team details for each player
team = pd.read_csv('data/nba_2017_att_val_elo.csv')
cols =  ['TOTAL', 'AVG', 'PCT', 'VALUE_MILLIONS', 'ELO']
teamDetails  = pd.DataFrame(columns = cols)
for i in range(0, data.shape[0]):
    playerTeam = data['TEAM'][i]
    index = -1
    j = 0
    while(index == -1 and j < team.shape[0]):
        if(team['TEAM'][j] == playerTeam):
            index = j
            x = team.iloc[[j]]
            x = x.drop(x.columns[0:3], axis = 1) 
            x = x.drop(x.columns[-1], axis = 1) 
            frames = [teamDetails, x]
            teamDetails = pd.concat(frames)
        j += 1


# merge salary, team details into main dataframe
teamDetails.index = range(teamDetails.shape[0])
numpySalaries = np.asarray(salaries)
data["SALARY"] = numpySalaries
data = pd.concat([data, teamDetails], axis=1)


# encode position details
for i in range(0, data.shape[0]):
    pos = data['POSITION'][i]
    if(pos == "PG"):
        data.at[i, 'POSITION'] = 1
    elif(pos == "SG"):
        data.at[i, 'POSITION'] = 2
    elif(pos == "SF"):
        data.at[i, 'POSITION'] = 3
    elif(pos == "PF"):
        data.at[i, 'POSITION'] = 4
    elif(pos == "C" or pos == "PF-C"):
        data.at[i, 'POSITION'] = 5


# remove unnecessary columns
data = data.drop('TEAM', 1)

# write new panda dataset to file
data.to_csv("dataset.csv", index=False)

