# Import libraries here later when you need them
import matplotlib
import pandas as pd
$pip install -U scikit-learn

LeagueofLegends.csv
_Columns.csv
_LeagueofLegends.csv
banValues.csv
deathValues.csv
goldValues.csv
objValues.csv

# Data info
df_columns = pd.read_csv('../input/_Columns.csv',sep=',')
df_raw = pd.read_csv('../input/_LeagueofLegends.csv',sep=',')
df_raw.info()

# Instead of just being a "number runner", the machine should be able to weigh situations based not on just the numbers, 
# but on events that have taken place in the game also. ex. Training the machine: winrate of team A and team B are both 50/50, 
# but I have told the machine that team A has gotten X objectives and has won. The machine should not just look at a winrate of 
# 80/20 and say that the team with 80% winrate will win, but look at the objectives/champions played and guess also.
# Will use professional games as examples as well as training, as well as my own personal games. I might even run the program while I'm playing
# to see if it works on the fly.

# Removed calculator, because winrates are not going to mean anything.
# I realized that what will be more telling of the machine's capabilities as a guesser will be its ability to extract conclusions from data that
# isn't just winrate.

