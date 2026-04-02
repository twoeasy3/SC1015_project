import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import os
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split
from PIL import Image

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)
st.title("Analysis of player valuation")

@st.cache_data
def load_data():
    csvFileLocation = "./data/merged_data.csv"
    data = pd.read_csv(csvFileLocation)
    
    # 1. Define conditions clearly
    # We use .fillna(False) to ensure no NaN values break the boolean logic
    conditions = [
        data['Pos'].str.contains("FW", na=False),
        data['Pos'].str.contains("MF", na=False),
        data['Pos'].str.contains("DF", na=False)
    ]
    posNames = ["Forward", "Midfielder", "Defender"]
    
    # 2. Generate gPos with a default value to avoid empty matches
    gPos = np.select(conditions, posNames, default="Other")

    # 3. Create encoding. Convert gPos to a Series first for better pd.get_dummies support
    new_var = pd.get_dummies(pd.Series(gPos), drop_first=True)
    
    # Ensure indices match before joining
    new_var.index = data.index
    data = data.join(new_var)

    # 4. Prepare X and Y
    # Drop non-numeric and target columns
    X = data.drop(columns=["market_value_in_eur", "Pos", "Player", "Squad", "Comp"], errors='ignore').select_dtypes(include=[np.number])
    Y = data['market_value_in_eur']
    
    # Avoid log(0) errors
    Y_log = np.log(Y.replace(0, 1))

    X_train, _, Y_train, _ = train_test_split(X, Y_log, test_size=0.33, shuffle=True, random_state=0)
    regr = LinearRegression().fit(X_train, Y_train)

    return data, regr
df, model = load_data()
df = df.sort_values(['Comp','Squad','Player'],
              ascending = [True,True,True])
pLeague = st.selectbox("Sort by League", pd.unique(df["Comp"]))
pClub = st.selectbox("Sort by Club", pd.unique(df.loc[df['Comp'] == pLeague, 'Squad']))
player = st.selectbox("Select the player to analyze", pd.unique(df.loc[df['Squad'] == pClub, 'Player']))
placeholder = st.empty()

if not player:
    st.error("Please select a player", icon="🚨")
else:
    st.write("### Input data for", player)

    # Extract player data
    df["col"] = df["Player"]==player
    player_df = df[df["col"]==1]
    # Plot all players
    st.write(player_df)
    ##Image file paths
    imagePaths = ["","",""] ##list of image paths
    imagePaths[0] = './data/images/clubs/' + player_df.loc[df['Player'] == player, 'Squad'].item() + ".png"
    imagePaths[1] = './data/images/leagues/' + player_df.loc[df['Player'] == player, 'Comp'].item() + ".png"
    imagePaths[2] = './data/images/players/' + player_df.loc[df['Player'] == player, 'Player'].item() + ".png"
    ##If player image does not exist, default image
    for i in range(0,3):
        if os.path.isfile(imagePaths[i]):
            pass
        else:
            imagePaths[i] = './data/images/players/!noImage.png'
    
    
    logo = Image.open(imagePaths[0])
    league = Image.open(imagePaths[1])
    playerImage = Image.open(imagePaths[2])
    st.image([playerImage,logo,league,],width=100)

    stat = st.selectbox("Select the statistic to analyze", options = ["MP","Age","Starts","Min","90s","Goals","Shots","SoT","SoT%","G/Sh","G/SoT","ShoDist",
                                                               "ShoFK","ShoPK","PKatt","PasTotCmp","PasTotAtt","PasTotCmp%","PasTotDist",
                                                               "PasTotPrgDist","PasShoCmp","PasShoAtt","PasShoCmp%","PasMedCmp","PasMedAtt",
                                                               "PasMedCmp%","PasLonCmp","PasLonAtt","PasLonCmp%","Assists","PasAss","Pas3rd",
                                                               "PPA","CrsPA","PasProg","PasAtt","PasLive","PasDead","PasFK","TB","PasPress","Sw",
                                                               "PasCrs","CK","CkIn","CkOut","CkStr","PasGround","PasLow","PasHigh","PaswLeft",
                                                               "PaswRight","PaswHead","TI","PaswOther","PasCmp","PasOff","PasOut","PasInt","PasBlocks",
                                                               "SCA","ScaPassLive","ScaPassDead","ScaDrib","ScaSh","ScaFld","ScaDef","GCA",
                                                               "GcaPassLive","GcaPassDead","GcaDrib","GcaSh","GcaFld","GcaDef","Tkl","TklWon",
                                                               "TklDef3rd","TklMid3rd","TklAtt3rd","TklDri","TklDriAtt","TklDri%","TklDriPast",
                                                               "Press","PresSucc","Press%","PresDef3rd","PresMid3rd","PresAtt3rd","Blocks","BlkSh",
                                                               "BlkShSv","BlkPass","Int","Tkl+Int","Clr","Err","Touches","TouDefPen","TouDef3rd",
                                                               "TouMid3rd","TouAtt3rd","TouAttPen","TouLive","DriSucc","DriAtt","DriSucc%","DriPast",
                                                               "DriMegs","Carries","CarTotDist","CarPrgDist","CarProg","Car3rd","CPA","CarMis","CarDis",
                                                               "RecTarg","Rec","Rec%","RecProg","CrdY","CrdR","2CrdY","Fls","Fld","Off","Crs","TklW",
                                                               "PKwon","PKcon","OG","Recov","AerWon","AerLost","AerWon%"])

    scope = st.selectbox("Select the scope to analyze", options = ["All", "Same League", "Same Club"])
    if scope == "All":
        st.write("Comparison to all players in database")
        fig = px.scatter(df,
                        x = stat, 
                        y = "market_value_in_eur", 
                        hover_name = "Player", 
                        color = "col",
                        labels = {'col': 'Choosen player:', 'market_value_in_eur': 'Market Value (EUR)'},
                        color_discrete_sequence=px.colors.qualitative.G10)
        st.plotly_chart(fig)
        percentile_rank = round((df[df[stat] <= (player_df.loc[df['Player'] == player, stat].item())].size / df.size) * 100,2)
        st.write("{0} is in the {1}th percentile in {2}".format(player,percentile_rank,stat))
        
    if scope == "Same League":
        st.write("Comparison to all players in", pLeague)
        fig = px.scatter(df.loc[df['Comp'] == pLeague],
                        x = stat, 
                        y = "market_value_in_eur", 
                        hover_name = "Player", 
                        color = "col",
                        labels = {'col': 'Choosen player:', 'market_value_in_eur': 'Market Value (EUR)'},
                        color_discrete_sequence=px.colors.qualitative.G10)
        st.plotly_chart(fig)
        percentile_rank = round((df.loc[df['Comp'] == pLeague][df.loc[df['Comp'] == pLeague][stat] <= (player_df.loc[df['Player'] == player, stat].item())].size / df.loc[df['Comp'] == pLeague].size) * 100,2)
        st.write("{0} is in the {1}th percentile in {2}".format(player,percentile_rank,stat))
        
    if scope == "Same Club":
        st.write("Comparison to all players in", pClub, "," , pLeague)
        fig = px.scatter(df.loc[df['Squad'] == pClub],
                        x = stat, 
                        y = "market_value_in_eur", 
                        hover_name = "Player", 
                        color = "col",
                        labels = {'col': 'Choosen player:', 'market_value_in_eur': 'Market Value (EUR)'},
                        color_discrete_sequence=px.colors.qualitative.G10)
        st.plotly_chart(fig)
        percentile_rank = round((df.loc[df['Squad'] == pClub][df.loc[df['Squad'] == pClub][stat] <= (player_df.loc[df['Player'] == player, stat].item())].size / df.loc[df['Squad'] == pClub].size) * 100,2)
        st.write("{0} is in the {1}th percentile in {2}".format(player,percentile_rank,stat))
    
    st.write("### Model value prediction for", player)
        # Ask the model for the exact columns it was trained on
    training_features = model.feature_names_in_
    
    # Filter the player's dataframe to only include those exact columns
    pred_df = player_df[training_features]
    pred_value = np.exp(model.predict(pred_df).item(0))
    act_value = player_df["market_value_in_eur"].values.item(0)
    diff_pct = 100*(act_value/pred_value - 1)
    st.write("Estimated player value is", str(round(pred_value/(10**6),2)), "MEUR.")
    st.write("Actual player value is", str(round(act_value/(10**6),2)), "MEUR.")
    if act_value > pred_value:
        st.write("According to the model the  player is", str(round(diff_pct)), "percent overvalued.")
    else: 
        st.write("According to the model the  player is", str(round(-diff_pct)), "percent undervalued.")
    
