import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import os
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split
import base64

st.set_page_config(
    page_title="Valuation Comparison",
    page_icon="✅",
    layout="wide",
)
st.title("Valuation comparison")

@st.cache_data 
def load_data():
    data = pd.read_csv("./data/merged_data.csv")
    
    # FIXED: Modernized np.select logic
    conditions = [
        data['Pos'].str.contains("FW", na=False),
        data['Pos'].str.contains("MF", na=False),
        data['Pos'].str.contains("DF", na=False)
    ]
    posNames = ["Forward", "Midfielder", "Defender"]
    gPos = np.select(conditions, posNames, default="Other")

    # FIXED: Convert to Series before get_dummies
    new_var = pd.get_dummies(pd.Series(gPos), drop_first=True)
    new_var.index = data.index
    data = data.join(new_var)

    # FIXED: Drop categorical/text columns safely, handle log(0)
    X = data.drop(columns=["market_value_in_eur", "Pos", "Player", "Squad", "Comp", "name"], errors='ignore').select_dtypes(include=[np.number])
    Y = data['market_value_in_eur']
    Y_log = np.log(Y.replace(0, 1))

    X_train, _, Y_train, _ = train_test_split(X, Y_log, test_size=0.33, shuffle=True, random_state=0)
    regr = LinearRegression().fit(X_train, Y_train)
    
    pred_Y_log = regr.predict(X)
    pred_Y = np.exp(pred_Y_log)
    
    return data, regr, pred_Y, Y

df, model, pred_Y, Y = load_data()
df = df.sort_values(['Comp','Squad','Player'], ascending=[True,True,True])

# Assuming "name" exists in your CSV. If it doesn't, change "name" to "Player"
info = pd.DataFrame(df[["Player", 'Comp', 'Squad','Pos']])
pred_Y = pd.DataFrame(pred_Y, columns=["predicted_market_value_in_eur"])

# Reset index on both before joining to ensure they align perfectly
comparison = info.reset_index(drop=True).join(Y.reset_index(drop=True)).join(pred_Y).round()
comparison["Undervalued(+) / Overvalued(-)"] = round((comparison["predicted_market_value_in_eur"]/comparison["market_value_in_eur"] - 1)*100, 2)

st.write(comparison)

# ==========================================
# Best Value XI
# ==========================================
st.write("### Best Value XI")
st.write("Picking the most undervalued players in a standard 4-4-2 formation, our algorithm deems this squad to be the best value for money")

xCoord = [0.25,0.5,0,0.25,0.5,0.75,0,0.25,0.5,0.75,0.375]
yCoord = [1,1,0.75,0.75,0.75,0.75,0.5,0.5,0.5,0.5,0.25]
sort_order = ["FW","FWMF","FWDF","MF","MFFW","MFDF","DF","DFMF","DFFW","GK"]

bargainEleven = ["Luigi Samele","Luca Wollschläger","Pedro Ortiz","Cristian Volpato","Evan Ferguson",
"Manu García","Sergio Santos","Kike Hermoso","Óscar Gil","Julian Eitschberger","Davide Marfella"]

selectedPlayers = df['Player'].isin(bargainEleven)
# FIXED: Added .copy() to prevent SettingWithCopyWarning
df11 = df[selectedPlayers].copy()
df11['Pos'] = pd.Categorical(df11['Pos'], categories=sort_order, ordered=True)
df11 = df11.sort_values('Pos')
df11 = df11.assign(xCoord=xCoord, yCoord=yCoord)

fig = px.scatter(df11, x='xCoord', y='yCoord', hover_name="Player", width=600, height=800, opacity=0,
                 hover_data={'xCoord':False,'yCoord':False,'Squad':True,'Comp':True},
                 color_discrete_sequence=px.colors.qualitative.G10)

fig.update_layout(coloraxis_showscale=False, xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False))
fig.update_xaxes(showticklabels=False, range=[-0.125,0.875], title_text='')
fig.update_yaxes(showticklabels=False, range=[0.110,1.125], title_text='')
fig.update_traces(marker={'size': 100})

# Load Pitch
if os.path.isfile("./data/images/data/pitch.jpg"):
    pitch = base64.b64encode(open("./data/images/data/pitch.jpg",'rb').read())
    fig.update_layout(images=[dict(source='data:image/jpg;base64,{}'.format(pitch.decode()), xref="paper", yref="paper",
                        x=0, y=1, sizex=1, sizey=1, xanchor="left", yanchor="top", sizing="stretch", layer="below")])

# FIXED: Added Fallback for Missing Player Images
for index, row in df11.iterrows():
    imageLocation = './data/images/players/' + row['Player'] + '.png'
    if not os.path.isfile(imageLocation):
        imageLocation = './data/images/players/!noImage.png'
        
    if os.path.isfile(imageLocation): # Double check even the fallback exists
        imageEncode = base64.b64encode(open(imageLocation,'rb').read())
        fig.add_layout_image(dict(source='data:image/png;base64,{}'.format(imageEncode.decode()),
                                x=row['xCoord']+0.125, y=row['yCoord']-0.1, xref='paper', yref='paper',
                                sizex=0.20, sizey=0.20, xanchor="center", yanchor="middle", layer="above"))

st.plotly_chart(fig)

# ==========================================
# Overvalued XI
# ==========================================
st.write("### Overvalued XI")
st.write("Our algorithm picked out this team to command transfer fees far beyond their recent contributions on the pitch.")

valueEleven = ['Liam Delap','Wesley Moraes','Kalvin Phillips','Wilfred Ndidi','Gianluca Gaetano',
'Federico Chiesa','Leonardo Spinazzola','Wesley Fofana','James Justin','Ben Chilwell','Sergio Rico']

selectedPlayers = df['Player'].isin(valueEleven)
# FIXED: Added .copy() to prevent SettingWithCopyWarning
df11o = df[selectedPlayers].copy()
df11o['Pos'] = pd.Categorical(df11o['Pos'], categories=sort_order, ordered=True)
df11o = df11o.sort_values('Pos')
df11o = df11o.assign(xCoord=xCoord, yCoord=yCoord)

fig2 = px.scatter(df11o, x='xCoord', y='yCoord', hover_name="Player", width=600, height=800, opacity=0,
                 hover_data={'xCoord':False,'yCoord':False,'Squad':True,'Comp':True},
                 color_discrete_sequence=px.colors.qualitative.G10)

fig2.update_layout(coloraxis_showscale=False, xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False))
fig2.update_xaxes(showticklabels=False, range=[-0.125,0.875], title_text='')
fig2.update_yaxes(showticklabels=False, range=[0.110,1.125], title_text='')
fig2.update_traces(marker={'size': 100})

# Load Pitch for Figure 2
if os.path.isfile("./data/images/data/pitch.jpg"):
    fig2.update_layout(images=[dict(source='data:image/jpg;base64,{}'.format(pitch.decode()), xref="paper", yref="paper",
                        x=0, y=1, sizex=1, sizey=1, xanchor="left", yanchor="top", sizing="stretch", layer="below")])

# FIXED: Added Fallback for Missing Player Images
for index, row in df11o.iterrows():
    imageLocation = './data/images/players/' + row['Player'] + '.png'
    if not os.path.isfile(imageLocation):
        imageLocation = './data/images/players/!noImage.png'
        
    if os.path.isfile(imageLocation):
        imageEncode = base64.b64encode(open(imageLocation,'rb').read())
        fig2.add_layout_image(dict(source='data:image/png;base64,{}'.format(imageEncode.decode()),
                                x=row['xCoord']+0.125, y=row['yCoord']-0.1, xref='paper', yref='paper',
                                sizex=0.20, sizey=0.20, xanchor="center", yanchor="middle", layer="above"))

st.plotly_chart(fig2)
