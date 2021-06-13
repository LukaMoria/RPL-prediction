
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import collections
import pickle
from flask import Flask, abort, jsonify, request
from flask_cors import CORS


# In[2]:


data = pd.read_csv("./data/RPL.csv", encoding = 'cp1251', delimiter=';')
RPL_2020_2021 = pd.read_csv('./data/teams 20-21.csv', encoding = 'cp1251')
teamList = RPL_2020_2021['Team Name'].tolist()

deleteTeam = [x for x in pd.unique(data['Соперник']) if x not in teamList]
for name in deleteTeam:
    data = data[data['Команда'] != name]
    data = data[data['Соперник'] != name]
data = data.reset_index(drop=True)

linear_regression_model = pickle.load(open('./models/linear_regression_model.pkl', 'rb'))


# In[3]:


#return season team statistics  
def GetSeasonTeamStat(team, season):
    goalScored = 0 #Голов забито
    goalAllowed = 0 #Голов пропущено

    gameWin = 0 #Выиграно
    gameDraw = 0 #Ничья
    gameLost = 0 #Проиграно

    totalScore = 0 #Количество набранных очков

    matches = 0 #Количество сыгранных матчей
    
    xG = 0 #Ожидаемые голы
    
    shot = 0 #Удары
    shotOnTarget = 0 #Удары в створ
    
    cross = 0 #Навесы
    accurateCross = 0 #Точные навесы
    
    totalHandle = 0 #Владение мячом
    averageHandle = 0 #Среднее владение мячом за матч
    
    Pass = 0 #Пасы
    accuratePass = 0 #Точные пасы
    
    PPDA = 0 #Интенсивность прессинга в матче

    for i in range(len(data)):
        if (((data['Год'][i] == season) and (data['Команда'][i] == team) and (data['Часть'][i] == 2)) or ((data['Год'][i] == season-1) and (data['Команда'][i] == team) and (data['Часть'][i] == 1))):
            matches += 1
                
            goalScored += data['Забито'][i]
            goalAllowed += data['Пропущено'][i]

            if (data['Забито'][i] > data['Пропущено'][i]):
                totalScore += 3
                gameWin += 1
            elif (data['Забито'][i] < data['Пропущено'][i]):
                gameLost +=1
            else:
                totalScore += 1
                gameDraw += 1
            
            xG += data['xG'][i]
            
            shot += data['Удары'][i]
            shotOnTarget += data['Удары в створ'][i]
            
            Pass += data['Передачи'][i]
            accuratePass += data['Точные передачи'][i]
            
            totalHandle += data['Владение'][i]
            
            cross += data['Навесы'][i]
            accurateCross += data['Точные навесы'][i]
            
            PPDA += data['PPDA'][i]

    averageHandle = round(totalHandle/matches, 3) #Владение мячом в среднем за матч
    
    return [gameWin, gameDraw, gameLost, 
            goalScored, goalAllowed, totalScore, 
            round(xG, 3), round(PPDA, 3),
            shot, shotOnTarget, 
            Pass, accuratePass,
            cross, accurateCross,
            round(averageHandle, 3)]


# In[4]:

def createGamePrediction(team1_vector, team2_vector):
    diff = [[a - b for a, b in zip(team1_vector, team2_vector)]]
    predictions = linear_regression_model.predict(diff)
    return predictions


# In[6]:


app = Flask(__name__)
CORS(app)

@app.route('/api/predictResult', methods=['POST']) #прочитать боди
def make_single_game_predict():
    data = request.get_json(force=True)
    team1_name = data['team1_name']
    team2_name = data['team2_name']
    team1_vector = GetSeasonTeamStat(team1_name, 2019)
    team2_vector = GetSeasonTeamStat(team2_name, 2019)
    output = {
      "team1": {
        "name": team1_name,
        "win_probability": createGamePrediction(team1_vector, team2_vector)[0]
      }, 
      "team2": {
        "name": team2_name,
        "win_probability": createGamePrediction(team2_vector, team1_vector)[0]
      }
    }
    print(output)
    return jsonify(results=output)

