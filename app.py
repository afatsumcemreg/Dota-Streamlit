import pandas as pd
import joblib
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_extras.no_default_selectbox import selectbox

from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

dota_image = Image.open("kisspng-dota-2-counter-strike-global-offensive-defense-of-5afd3c9e8ce7c6.9674077815265455665772.png")

st.set_page_config(
    page_title="Dota 2 Win Simulation",
    page_icon=dota_image,
    layout="wide",
    initial_sidebar_state="expanded",
)

from sklearn.base import BaseEstimator, TransformerMixin
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


def Dota():

    
    st.markdown(f'<h1 <p><font face="tahoma" size="130" color="white"><b>    DotA-2 Win Simulation</b></font></p> </h1>', unsafe_allow_html=True)
   

    dsg = st.sidebar.selectbox('**Go to**', ["Introduction", "Predict Probability"])
    
    
    if dsg == "Intro":
    
         clm1, clm2, clm3 = st.columns([0.4, 0.1, 0.3])
         
         with clm1:
            
            st.image(dota_image, width=360, use_column_width=False, channels='RGB')
            
         with clm2:
            gorkem = Image.open("Gorkem_Aksoy.png")
            st.image(gorkem, width=62, use_column_width=False, channels='RGB')
            
            hakan = Image.open("Hakan_Yuceturk.png")
            st.image(hakan, width=62, use_column_width=False, channels='RGB')
            
            mustafa = Image.open("m_germec_photo.png")
            st.image(mustafa, width=64, use_column_width=False, channels='RGB')
            
            kerim = Image.open("Kerimcan_Arslan.png")
            st.image(kerim, width=66, use_column_width=False, channels='RGB')
            
            berkay = Image.open("Berkay_Basak.png")
            st.image(berkay, width=66, use_column_width=False, channels='RGB')
            
         with clm3:
            
            st.markdown(f'<p><font face="tahoma" size="5" color="white"><b>Görkem Aksoy</b></font></p>', unsafe_allow_html=True)
            st.markdown(f'<p><font face="tahoma" size="5" color="white"><b>  </b></font></p>', unsafe_allow_html=True)
            st.markdown(f'<p><font face="tahoma" size="5" color="white"><b>  </b></font></p>', unsafe_allow_html=True)
            
            st.markdown(f'<p><font face="tahoma" size="5" color="white"><b>Hakan Yücetürk</b></font></p>', unsafe_allow_html=True)
            st.markdown(f'<p><font face="tahoma" size="5" color="white"><b>  </b></font></p>', unsafe_allow_html=True)
            st.markdown(f'<p><font face="tahoma" size="5" color="white"><b>  </b></font></p>', unsafe_allow_html=True)
            
            st.markdown(f'<p><font face="tahoma" size="5" color="white"><b>Mustafa Germeç</b></font></p>', unsafe_allow_html=True)
            st.markdown(f'<p><font face="tahoma" size="5" color="white"><b>  </b></font></p>', unsafe_allow_html=True)
            st.markdown(f'<p><font face="tahoma" size="5" color="white"><b>  </b></font></p>', unsafe_allow_html=True)
            
            st.markdown(f'<p><font face="tahoma" size="5" color="white"><b>Kerimcan Arslan</b></font></p>', unsafe_allow_html=True)
            st.markdown(f'<p><font face="tahoma" size="5" color="white"><b>  </b></font></p>', unsafe_allow_html=True)
            st.markdown(f'<p><font face="tahoma" size="5" color="white"><b>  </b></font></p>', unsafe_allow_html=True)
            
            st.markdown(f'<p><font face="tahoma" size="5" color="white"><b>İsmail B. Başak</b></font></p>', unsafe_allow_html=True)
            
            

    else:
        col1, col2, col3, col4, col5 = st.columns([0.6, 1, 0.6, 1, 0.6], gap="small")
        
        with col2:
            
            hero=pd.read_csv("hero_list.csv")
            

            radiant_1 = st.selectbox("Radiant Takımı 1.Hero Seçimi", hero, index=5)
            radiant_2 = st.selectbox("Radiant Takımı 2.Hero Seçimi", hero, index=9)
            radiant_3 = st.selectbox("Radiant Takımı 3.Hero Seçimi", hero, index=12)
            radiant_4 = st.selectbox("Radiant Takımı 4.Hero Seçimi", hero, index=16)
            radiant_5 = st.selectbox("Radiant Takımı 5.Hero Seçimi", hero, index=57)
            
       
        with col1:
            image_r_1 = Image.open(f'dota_2_hero_resim/{radiant_1}.png')
            st.image(image_r_1, width=75, use_column_width=True, channels='RGB')
            
            image_r_2 = Image.open(f'dota_2_hero_resim/{radiant_2}.png')
            st.image(image_r_2, width=75, use_column_width=True, channels='RGB')
            
            image_r_3 = Image.open(f'dota_2_hero_resim/{radiant_3}.png')
            st.image(image_r_3, width=75, use_column_width=True, channels='RGB')
            
            image_r_4 = Image.open(f'dota_2_hero_resim/{radiant_4}.png')
            st.image(image_r_4, width=75, use_column_width=True, channels='RGB')
            
            image_r_5 = Image.open(f'dota_2_hero_resim/{radiant_5}.png')
            st.image(image_r_5, width=75, use_column_width=True, channels='RGB')

        with col4:
            dire_1 = st.selectbox("Dire Takımı 1.Hero Seçimi", hero, index=0)
            dire_2 = st.selectbox("Dire Takımı 2.Hero Seçimi", hero, index=1)
            dire_3 = st.selectbox("Dire Takımı 3.Hero Seçimi", hero, index=2)
            dire_4 = st.selectbox("Dire Takımı 4.Hero Seçimi", hero, index=3)
            dire_5 = st.selectbox("Dire Takımı 5.Hero Seçimi", hero, index=4)


        with col5:
            image_d_1 = Image.open(f'dota_2_hero_resim/{dire_1}.png')
            st.image(image_d_1, width=75, use_column_width=True, channels='RGB')
            
            image_d_2 = Image.open(f'dota_2_hero_resim/{dire_2}.png')
            st.image(image_d_2, width=75, use_column_width=True, channels='RGB')
            
            image_d_3 = Image.open(f'dota_2_hero_resim/{dire_3}.png')
            st.image(image_d_3, width=75, use_column_width=True, channels='RGB')
            
            image_d_4 = Image.open(f'dota_2_hero_resim/{dire_4}.png')
            st.image(image_d_4, width=75, use_column_width=True, channels='RGB')
            
            image_d_5 = Image.open(f'dota_2_hero_resim/{dire_5}.png')
            st.image(image_d_5, width=75, use_column_width=True, channels='RGB')



        with col3:

            def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
                dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
                return dataframe
            

            data_hero_roles = pd.read_csv("data_hero_roles.csv")

            feature_to_use_ = pd.read_csv("Oyun_Oncesi_Feature_Names.csv")

            df = pd.read_csv("df.csv")

            model = joblib.load("dota_oyun_oncesi.pkl")

            player_heroes_oyun_oncesi = ['Abaddon', 'Alchemist', 'Ancient Apparition', 'Anti-Mage', 'Axe',
                'Bane', 'Batrider', 'Beastmaster', 'Bloodseeker', 'Bounty Hunter',
                'Brewmaster', 'Bristleback', 'Broodmother', 'Centaur Warrunner',
                'Chaos Knight', 'Chen', 'Clinkz', 'Clockwerk', 'Crystal Maiden',
                'Dark Seer', 'Dazzle', 'Death Prophet', 'Disruptor', 'Doom',
                'Dragon Knight', 'Drow Ranger', 'Earth Spirit', 'Earthshaker',
                'Elder Titan', 'Ember Spirit', 'Enchantress', 'Enigma',
                'Faceless Void', 'Gyrocopter', 'Huskar', 'Invoker', 'Io', 'Jakiro',
                'Juggernaut', 'Keeper of the Light', 'Kunkka', 'Legion Commander',
                'Leshrac', 'Lich', 'Lifestealer', 'Lina', 'Lion', 'Lone Druid',
                'Luna', 'Lycan', 'Magnus', 'Medusa', 'Meepo', 'Mirana',
                'Morphling', 'Naga Siren', "Nature's Prophet", 'Necrophos',
                'Night Stalker', 'Nyx Assassin', 'Ogre Magi', 'Omniknight',
                'Oracle', 'Outworld Devourer', 'Phantom Assassin',
                'Phantom Lancer', 'Phoenix', 'Puck', 'Pudge', 'Pugna',
                'Queen of Pain', 'Razor', 'Riki', 'Rubick', 'Sand King',
                'Shadow Demon', 'Shadow Fiend', 'Shadow Shaman', 'Silencer',
                'Skywrath Mage', 'Slardar', 'Slark', 'Sniper', 'Spectre',
                'Spirit Breaker', 'Storm Spirit', 'Sven', 'Techies',
                'Templar Assassin', 'Terrorblade', 'Tidehunter', 'Timbersaw',
                'Tinker', 'Tiny', 'Treant Protector', 'Troll Warlord', 'Tusk',
                'Undying', 'Ursa', 'Vengeful Spirit', 'Venomancer', 'Viper',
                'Visage', 'Warlock', 'Weaver', 'Windranger', 'Winter Wyvern',
                'Witch Doctor', 'Wraith King', 'Zeus']
           
            h_roles_oyun_oncesi = ['Carries', 'Supports', 'Nukers', 'Disablers', 'Junglers', 'Durable', 'Escape', 'Pushers','Initiators']
            h_specs_oyun_oncesi = ['Strength', 'Agility', 'Intelligence', 'Health_avg', 'Health_reg_avg', 'Mana_avg', 'Mana_reg_avg', 'Armor_avg', 'Att/sec_avg', 'Damage_avg', 'Magic resistance', 'Movement speed', 'Attack speed', 'Turn rate', 'Attack range', 'Damage block']

            radiant_cols_oyun_oncesi = ['radiant_Abaddon',
             'radiant_Alchemist',
             'radiant_Ancient Apparition',
             'radiant_Anti-Mage',
             'radiant_Axe',
             'radiant_Bane',
             'radiant_Batrider',
             'radiant_Beastmaster',
             'radiant_Bloodseeker',
             'radiant_Bounty Hunter',
             'radiant_Brewmaster',
             'radiant_Bristleback',
             'radiant_Broodmother',
             'radiant_Centaur Warrunner',
             'radiant_Chaos Knight',
             'radiant_Chen',
             'radiant_Clinkz',
             'radiant_Clockwerk',
             'radiant_Crystal Maiden',
             'radiant_Dark Seer',
             'radiant_Dazzle',
             'radiant_Death Prophet',
             'radiant_Disruptor',
             'radiant_Doom',
             'radiant_Dragon Knight',
             'radiant_Drow Ranger',
             'radiant_Earth Spirit',
             'radiant_Earthshaker',
             'radiant_Elder Titan',
             'radiant_Ember Spirit',
             'radiant_Enchantress',
             'radiant_Enigma',
             'radiant_Faceless Void',
             'radiant_Gyrocopter',
             'radiant_Huskar',
             'radiant_Invoker',
             'radiant_Io',
             'radiant_Jakiro',
             'radiant_Juggernaut',
             'radiant_Keeper of the Light',
             'radiant_Kunkka',
             'radiant_Legion Commander',
             'radiant_Leshrac',
             'radiant_Lich',
             'radiant_Lifestealer',
             'radiant_Lina',
             'radiant_Lion',
             'radiant_Lone Druid',
             'radiant_Luna',
             'radiant_Lycan',
             'radiant_Magnus',
             'radiant_Medusa',
             'radiant_Meepo',
             'radiant_Mirana',
             'radiant_Morphling',
             'radiant_Naga Siren',
             "radiant_Nature's Prophet",
             'radiant_Necrophos',
             'radiant_Night Stalker',
             'radiant_Nyx Assassin',
             'radiant_Ogre Magi',
             'radiant_Omniknight',
             'radiant_Oracle',
             'radiant_Outworld Devourer',
             'radiant_Phantom Assassin',
             'radiant_Phantom Lancer',
             'radiant_Phoenix',
             'radiant_Puck',
             'radiant_Pudge',
             'radiant_Pugna',
             'radiant_Queen of Pain',
             'radiant_Razor',
             'radiant_Riki',
             'radiant_Rubick',
             'radiant_Sand King',
             'radiant_Shadow Demon',
             'radiant_Shadow Fiend',
             'radiant_Shadow Shaman',
             'radiant_Silencer',
             'radiant_Skywrath Mage',
             'radiant_Slardar',
             'radiant_Slark',
             'radiant_Sniper',
             'radiant_Spectre',
             'radiant_Spirit Breaker',
             'radiant_Storm Spirit',
             'radiant_Sven',
             'radiant_Techies',
             'radiant_Templar Assassin',
             'radiant_Terrorblade',
             'radiant_Tidehunter',
             'radiant_Timbersaw',
             'radiant_Tinker',
             'radiant_Tiny',
             'radiant_Treant Protector',
             'radiant_Troll Warlord',
             'radiant_Tusk',
             'radiant_Undying',
             'radiant_Ursa',
             'radiant_Vengeful Spirit',
             'radiant_Venomancer',
             'radiant_Viper',
             'radiant_Visage',
             'radiant_Warlock',
             'radiant_Weaver',
             'radiant_Windranger',
             'radiant_Winter Wyvern',
             'radiant_Witch Doctor',
             'radiant_Wraith King',
             'radiant_Zeus']

            dire_cols_oyun_oncesi = ['dire_Abaddon',
             'dire_Alchemist',
             'dire_Ancient Apparition',
             'dire_Anti-Mage',
             'dire_Axe',
             'dire_Bane',
             'dire_Batrider',
             'dire_Beastmaster',
             'dire_Bloodseeker',
             'dire_Bounty Hunter',
             'dire_Brewmaster',
             'dire_Bristleback',
             'dire_Broodmother',
             'dire_Centaur Warrunner',
             'dire_Chaos Knight',
             'dire_Chen',
             'dire_Clinkz',
             'dire_Clockwerk',
             'dire_Crystal Maiden',
             'dire_Dark Seer',
             'dire_Dazzle',
             'dire_Death Prophet',
             'dire_Disruptor',
             'dire_Doom',
             'dire_Dragon Knight',
             'dire_Drow Ranger',
             'dire_Earth Spirit',
             'dire_Earthshaker',
             'dire_Elder Titan',
             'dire_Ember Spirit',
             'dire_Enchantress',
             'dire_Enigma',
             'dire_Faceless Void',
             'dire_Gyrocopter',
             'dire_Huskar',
             'dire_Invoker',
             'dire_Io',
             'dire_Jakiro',
             'dire_Juggernaut',
             'dire_Keeper of the Light',
             'dire_Kunkka',
             'dire_Legion Commander',
             'dire_Leshrac',
             'dire_Lich',
             'dire_Lifestealer',
             'dire_Lina',
             'dire_Lion',
             'dire_Lone Druid',
             'dire_Luna',
             'dire_Lycan',
             'dire_Magnus',
             'dire_Medusa',
             'dire_Meepo',
             'dire_Mirana',
             'dire_Morphling',
             'dire_Naga Siren',
             "dire_Nature's Prophet",
             'dire_Necrophos',
             'dire_Night Stalker',
             'dire_Nyx Assassin',
             'dire_Ogre Magi',
             'dire_Omniknight',
             'dire_Oracle',
             'dire_Outworld Devourer',
             'dire_Phantom Assassin',
             'dire_Phantom Lancer',
             'dire_Phoenix',
             'dire_Puck',
             'dire_Pudge',
             'dire_Pugna',
             'dire_Queen of Pain',
             'dire_Razor',
             'dire_Riki',
             'dire_Rubick',
             'dire_Sand King',
             'dire_Shadow Demon',
             'dire_Shadow Fiend',
             'dire_Shadow Shaman',
             'dire_Silencer',
             'dire_Skywrath Mage',
             'dire_Slardar',
             'dire_Slark',
             'dire_Sniper',
             'dire_Spectre',
             'dire_Spirit Breaker',
             'dire_Storm Spirit',
             'dire_Sven',
             'dire_Techies',
             'dire_Templar Assassin',
             'dire_Terrorblade',
             'dire_Tidehunter',
             'dire_Timbersaw',
             'dire_Tinker',
             'dire_Tiny',
             'dire_Treant Protector',
             'dire_Troll Warlord',
             'dire_Tusk',
             'dire_Undying',
             'dire_Ursa',
             'dire_Vengeful Spirit',
             'dire_Venomancer',
             'dire_Viper',
             'dire_Visage',
             'dire_Warlock',
             'dire_Weaver',
             'dire_Windranger',
             'dire_Winter Wyvern',
             'dire_Witch Doctor',
             'dire_Wraith King',
             'dire_Zeus']
             
            radiant_roles_oyun_oncesi = ['radiant_Carries',
             'radiant_Supports',
             'radiant_Nukers',
             'radiant_Disablers',
             'radiant_Junglers',
             'radiant_Durable',
             'radiant_Escape',
             'radiant_Pushers',
             'radiant_Initiators']

            dire_roles_oyun_oncesi = ['dire_Carries',
             'dire_Supports',
             'dire_Nukers',
             'dire_Disablers',
             'dire_Junglers',
             'dire_Durable',
             'dire_Escape',
             'dire_Pushers',
             'dire_Initiators']

            radiant_h_specs_oyun_oncesi = ['radiant_Strength',
             'radiant_Agility',
             'radiant_Intelligence',
             'radiant_Health_avg',
             'radiant_Health_reg_avg',
             'radiant_Mana_avg',
             'radiant_Mana_reg_avg',
             'radiant_Armor_avg',
             'radiant_Att/sec_avg',
             'radiant_Damage_avg',
             'radiant_Magic resistance',
             'radiant_Movement speed',
             'radiant_Attack speed',
             'radiant_Turn rate',
             'radiant_Attack range',
             'radiant_Damage block']

            dire_h_specs_oyun_oncesi = ['dire_Strength',
             'dire_Agility',
             'dire_Intelligence',
             'dire_Health_avg',
             'dire_Health_reg_avg',
             'dire_Mana_avg',
             'dire_Mana_reg_avg',
             'dire_Armor_avg',
             'dire_Att/sec_avg',
             'dire_Damage_avg',
             'dire_Magic resistance',
             'dire_Movement speed',
             'dire_Attack speed',
             'dire_Turn rate',
             'dire_Attack range',
             'dire_Damage block']

            def prepare_dataframe(heroes_radiant, heroes_dire, feature_names):
                # Heroes, roles, specs
                X = None

                radiant_heroes = []
                dire_heroes = []
                radiant_roles_list = []
                dire_roles_list = []
                radiant_specs = []
                dire_specs = []
                all_heroes= []

                all_heroes.extend(heroes_radiant)
                all_heroes.extend(heroes_dire)

                new_hero_df = pd.DataFrame(all_heroes, columns = ["hero"])
                new_hero_df = pd.merge(new_hero_df, df, on ='hero', how ='left')

                for col in data_hero_roles.columns:
                    new_hero_df[col] = new_hero_df['hero'].apply(lambda _h_name: 1 if _h_name in data_hero_roles[col].values else 0)
                    
                new_hero_df = one_hot_encoder(new_hero_df, ["hero"], drop_first=False)
                new_hero_df.columns = [col.replace("hero_", "") for col in new_hero_df.columns]

                radi_hero = pd.DataFrame([0]*110, index=player_heroes_oyun_oncesi).T
                
                for col in heroes_radiant:
                    radi_hero.loc[:,col] = 1

                dire_hero = pd.DataFrame([0]*110, index=player_heroes_oyun_oncesi).T
                
                for col in heroes_dire:
                    dire_hero.loc[:,col] = 1


                radiant_heroes.append(radi_hero.values)
                dire_heroes.append(dire_hero.values)
                radiant_roles_list.append(new_hero_df[h_roles_oyun_oncesi][:5].sum().values)
                dire_roles_list.append(new_hero_df[h_roles_oyun_oncesi][5:].sum().values)
                radiant_specs.append(new_hero_df[h_specs_oyun_oncesi][:5].sum().values)
                dire_specs.append(new_hero_df[h_specs_oyun_oncesi][5:].sum().values)

                radiant_heroes = pd.DataFrame(np.array(radiant_heroes).reshape(1,-1), columns=radiant_cols_oyun_oncesi).applymap(lambda x: 1 if x > 0 else 0)
                dire_heroes = pd.DataFrame(np.array(dire_heroes).reshape(1,-1), columns=dire_cols_oyun_oncesi).applymap(lambda x: 1 if x > 0 else 0)
                radiant_roles_list = pd.DataFrame(radiant_roles_list, columns=radiant_roles_oyun_oncesi)
                dire_roles_list = pd.DataFrame(dire_roles_list, columns=dire_roles_oyun_oncesi)
                radiant_specs = pd.DataFrame(radiant_specs, columns=radiant_h_specs_oyun_oncesi)
                dire_specs = pd.DataFrame(dire_specs, columns=dire_h_specs_oyun_oncesi)

                X = pd.concat([radiant_heroes, dire_heroes, radiant_roles_list, dire_roles_list, radiant_specs, dire_specs], axis=1)
                return X[feature_names["Feature_Name"].values]
            
            heroes_radiant=[radiant_1, radiant_2, radiant_3, radiant_4, radiant_5]
            heroes_dire=[dire_1, dire_2, dire_3, dire_4, dire_5]

            sample_ = prepare_dataframe(heroes_radiant, heroes_dire, feature_to_use_)
            win_probility = model.predict_proba(sample_)[:, 1] * 100
            win_ = model.predict(sample_)[0]
            win_probility = win_probility.round(2)[0]
            
            if win_==1:
                st.success(f'Radiant Win: {win_}')
                st.markdown(f'<h1 <p><font face="tahoma" size="4.5" color="lime"><b>Probability of Win: %{win_probility}</b></font></p> </h1>', unsafe_allow_html=True)
            else:
                st.error(f'Radiant Win: {win_}')
                st.markdown(f'<h1 <p><font face="tahoma" size="4.5" color="maroon"><b>Probability of Win: %{win_probility}</b></font></p> </h1>', unsafe_allow_html=True)
            
        wlb_df = pd.read_csv("wlb_hero_final.csv")

        wlb_df = wlb_df.set_index('index')
        wlb_df = wlb_df.iloc[:,1:]
        def recommend_dota_heroes(win_loss_rates_wlb=[], hero_to_counter=[], enemy_team=[], our_team=[],
                                  number_of_recommendations=5):
            if hero_to_counter not in wlb_df.columns:
                print("You entered wrong hero name")
            recommend = win_loss_rates_wlb.loc[hero_to_counter, :].dropna()
            all_hero = enemy_team
            all_hero.extend(our_team)
            return recommend[set(recommend.index).difference(set(all_hero))].sort_values()[:number_of_recommendations]

        
        st.markdown(f'<h1 <p><font face="tahoma" size="20" color="maroon"><b>Hero Seçim Tavsiyesi</b></font></p> </h1>', unsafe_allow_html=True)
        
        abc = st.selectbox("Hero Seçiniz", hero)
        hero_recommender = recommend_dota_heroes(wlb_df, abc, heroes_dire, heroes_radiant, 10)
        st.dataframe(hero_recommender)
                
        

def Prediction():

    sayfa = st.sidebar.selectbox('**Go to**', ["İtem Tavsiye Sistemi", "Canlı Tahmin Modeli"])

    if sayfa == "İtem Tavsiye Sistemi":
        
        
        rec_list = pd.read_csv("tavsiyeler.csv")
        rec_list.drop("Unnamed: 0", axis=1, inplace=True)
        rec_list.index = rec_list["Item_Names"]
        rec_list.drop("Item_Names", axis=1, inplace=True)
        hero_list = list(rec_list.columns)
        item_list = list(rec_list.index)
        hero_chosen = st.selectbox("Hero seçiniz", hero_list) #sidebar.selectbox ile yan tarafta kutucuk açılabilir
        #st.columns sayfayı bölüyor, col1, col2 = st.columns(2) sonrada with col1: diye başlıyoruz

        if hero_chosen:
            item_list_suggestions = []
            for index in range(0, len(rec_list.index)):
                if rec_list[hero_chosen][index] == rec_list[hero_chosen][index]:
                    item_list_suggestions.append(item_list[index])
            len(item_list_suggestions)
            item_chosen = st.selectbox("Eşya seçiniz", item_list_suggestions)

        if item_chosen:
            recommendation = rec_list.loc[rec_list.index == item_chosen,hero_chosen][0][1:-1]
            new_list = recommendation.split(",")
            fin_list = []
            for index, elem in enumerate(new_list):
                if index == 0:
                    new_elem = elem[1:-1]
                    fin_list.append(new_elem)
                else:
                    new_elem = elem[2:-1]
                    fin_list.append(new_elem)
            for elem in fin_list:
                st.write(elem)
                
    else:
        oyun_ici_feature_engineering_data = pd.read_csv("oyun_ici_feature_engineering_data.csv")
        num_cols = ['dire_gold_per_min',
                    'dire_hero_healing',
                    'dire_kills_div_dire_total_damage',
                    'dire_last_hits',
                    'dire_last_hits_div_dire_total_damage',
                    'dire_last_hits_div_radiant_total_damage',
                    'dire_stuns_div_radiant_total_damage',
                    'dire_stuns_mod_dire_total_damage',
                    'dire_total_damage_mod_radiant_total_damage',
                    'dire_total_gold_gained_killing',
                    'dire_unit_attack_orders',
                    'dire_unit_order_move_to_target',
                    'dire_xp_per_min',
                    'min_dire_gold_spent_dire_gold_buyback_dire_total_gold_gained_killing',
                    'min_radiant_gold_spent_radiant_gold_buyback_radiant_total_gold_gained_killing',
                    'radiant_assists_div_radiant_total_damage',
                    'radiant_hero_healing',
                    'radiant_kills_div_radiant_total_damage',
                    'radiant_last_hits_div_dire_total_damage',
                    'radiant_stuns_div_dire_total_damage',
                    'radiant_stuns_mod_dire_total_damage',
                    'radiant_total_damage_mod_dire_total_damage',
                    'radiant_unit_attack_orders',
                    'radiant_unit_order_move_to_target',
                    'radiant_xp_per_min']
                    
        pipe_from_joblib_oyun_ici = joblib.load('dota_oyun_ici (1).pkl')
        
        df_ici = oyun_ici_feature_engineering_data.copy()
        
        Match = st.selectbox("Match ID Seçiniz", range(1, 50))
        
        cl1, cl2, cl3, cl4 = st.columns(4)
        
        with cl1:
        
            st.markdown("10.Dakika")
            for col in num_cols:
                df_ici[col] = df_ici[col] / 4
                
            a = df_ici.iloc[[Match]]
            y_pred0 = pipe_from_joblib_oyun_ici.predict_proba(a)
            y_pred0 = y_pred0[0][1]*100
            if y_pred0 >= 50:
                st.success(f'Radiant %{y_pred0} ihtimalle kazanıyor')
            else:
                st.error(f'Radiant %{y_pred0} ihtimalle kazanıyor')

        #########
        
        with cl2:
        
            st.markdown("20.Dakika")
            df_ici1 = oyun_ici_feature_engineering_data.copy()
            for col in num_cols:
                df_ici1[col] = df_ici1[col] / 2
                
            b = df_ici1.iloc[[Match]]
            y_pred1 = pipe_from_joblib_oyun_ici.predict_proba(b)
            y_pred1 = y_pred1.round(4)[0][1]*100
            if y_pred1 >= 50:
                st.success(f'Radiant %{y_pred1} ihtimalle kazanıyor')
            else:
                st.error(f'Radiant %{y_pred1} ihtimalle kazanıyor')

        ###########
        
        with cl3:
        
            st.markdown("30.Dakika")
            df_ici2 = oyun_ici_feature_engineering_data.copy()
            for col in num_cols:
                df_ici2[col] = df_ici2[col] / 4*3
                
            c = df_ici2.iloc[[Match]]
            y_pred2 = pipe_from_joblib_oyun_ici.predict_proba(c)
            y_pred2 = y_pred2.round(3)[0][1]*100
            if y_pred2 >= 50:
                st.success(f'Radiant %{y_pred2} ihtimalle kazanıyor')
            else:
                st.error(f'Radiant %{y_pred2} ihtimalle kazanıyor')

        ###########
        
        with cl4:
          
            st.markdown("Maç Sonucu")
            df_ici3 = oyun_ici_feature_engineering_data.copy()
                
            d = df_ici3.iloc[[Match]]
            y_pred3 = pipe_from_joblib_oyun_ici.predict(d)
            y_pred3 = y_pred3[0]
            
            if y_pred3 == 1:
                st.success("Radiant Takımı Kazanıyor")
            else:
                st.error("Radiant) Takımı Kaybediyor")
        
                    
            
def NLP():



    # TURNING SIMPLE PYTHON SCRIPTS INTO WEB APPS
    st.title('DotA-2 Sentiment Modeling')
    st.sidebar.title('DotA-2 Sentiment Modeling')

    page = st.sidebar.selectbox('**Go to**', ["Segment Frequency", "Word Cloud", "Prediction"])

    if page == "Segment Frequency":
        # LOADING THE DOTA SENTIMENT DATA
        @st.cache(persist=True)
        def load_data(text):
            data = pd.read_csv(text)
            return data

        df = load_data('dota_w.csv')

        st.sidebar.subheader('Show random chats')
        random_chat = st.sidebar.radio('**Sentiment**', ('Positive', 'Negative'))
        st.sidebar.markdown(df.query('sentiment_label == @random_chat')[['key']].sample(n=1).iat[0, 0])

        # PLOT INTERACTIVE BAR PLOTS AND PIE CHARTS
        st.sidebar.markdown('### Number of chats by sentiment')
        select = st.sidebar.selectbox('**Visualization type**', ['Histogram', 'Pie Chart'], key='1')
        sentiment_count = df['sentiment_label'].value_counts()
        sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index,
                                        'Chats': sentiment_count.values})
        st.markdown('### Number of chats by sentiment')
        if select == 'Histogram':
            fig = px.bar(sentiment_count, x='Sentiment', y='Chats', color='Chats', height=500)
            st.plotly_chart(fig)
        else:
            fig = px.pie(sentiment_count, values='Chats', names='Chats', hover_name='Sentiment')
            st.plotly_chart(fig)

    elif page == 'Word Cloud':
        # LOADING THE DOTA SENTIMENT DATA
        @st.cache(persist=True)
        def load_data(text):
            data = pd.read_csv(text)
            return data

        df = load_data('dota_w.csv')

        # WORD CLOUD FOR POSITIVE AND NEGATIVE MEANING CHATS
        word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?',
                                          ('Positive', 'Negative'))
        st.header('Word cloud for %s sentiment' % (word_sentiment))
        df = df[df['sentiment_label'] == word_sentiment]
        words = " ".join(i for i in df['key'])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=440, width=800).generate(
            words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        st.pyplot()

    else:
        # LOADING THE DOTA SENTIMENT DATA
        st.sidebar.header('Prediction')

        # LOADING THE DOTA SENTIMENT DATA
        @st.cache(persist=True)
        def load_data(text):
            data = pd.read_csv(text)
            return data


        df_new = load_data('dota_ml.csv')

        # dataframe
        def preparation_for_modeling(dataframe):
            X_train, X_test, y_train, y_test = train_test_split(dataframe['key'], dataframe['sentiment_label'],
                                                                test_size=0.25, random_state=19)
            vectorizer = CountVectorizer()
            X_count = vectorizer.fit_transform(X_train)
            tf_idf_word_vectorizer = TfidfVectorizer()
            X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X_train)
            return X_train, y_train, X_count, X_tf_idf_word


        X_train, y_train, X_count, X_tf_idf_word = preparation_for_modeling(df_new)

        # Load the model
        model_count = joblib.load("voting_clf_count_1.pkl")


        def predict(inputs):
            prediction_count = model_count.predict(inputs)
            return prediction_count


        # Create a UI element to take input and show the prediction
        inputs = st.sidebar.text_input("Enter input value")
        inputs = pd.Series(inputs)
        inputs = CountVectorizer().fit(X_train).transform(inputs)

        prediction_count = predict(inputs)
        if prediction_count == 1:
            st.write("Prediction: ", 'CONGRATULATIONS! I THINK YOU WON THE GAME!')
        else:
            st.write("Prediction: ", 'SORRY! I THINK YOU LOST THE GAME!')




page_names_to_funcs = {
    "Dota": Dota,
    "Real Time Prediction": Prediction,
    "NLP": NLP
}

demo_name = st.sidebar.selectbox("Choose a section", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()



