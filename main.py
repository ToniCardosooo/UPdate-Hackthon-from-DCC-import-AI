import requests
from urllib.request import urlopen
import json
import datetime
import time
import pickle
import pandas as pd
from llm import generate_review, set_embeddings
from xgb import generate_model

generate_model()
df = pd.read_csv("toni.csv")
df_emb = df.head(1000)
set_embeddings(df_emb)

game_modes_look_up = {1: 'Single player', 2: 'Multiplayer', 3: 'Co-operative', 4: 'Split screen', 5: 'Massively Multiplayer Online (MMO)', 6: 'Battle Royale'}

genres_look_up = {4: 'Fighting', 5: 'Shooter', 7: 'Music', 8: 'Platform', 9: 'Puzzle', 10: 'Racing', 11: 'Real Time Strategy (RTS)', 12: 'Role-playing (RPG)', 13: 'Simulator', 14: 'Sport', 15: 'Strategy', 16: 'Turn-based strategy (TBS)', 24: 'Tactical', 26: 'Quiz/Trivia', 25: "Hack and slash/Beat 'em up", 30: 'Pinball', 31: 'Adventure', 33: 'Arcade', 34: 'Visual Novel', 32: 'Indie', 35: 'Card & Board Game', 36: 'MOBA', 2: 'Point-and-click'}

platforms_look_up = {158: 'Commodore CDTV', 339: 'Sega Pico', 8: 'PlayStation 2', 39: 'iOS', 94: 'Commodore Plus/4', 144: 'AY-3-8710', 88: 'Odyssey', 90: 'Commodore PET', 237: 'Sol-20', 6: 'PC (Microsoft Windows)', 44: 'Tapwave Zodiac', 129: 'Texas Instruments TI-99', 134: 'Acorn Electron', 378: 'Gamate', 135: 'Hyper Neo Geo 64', 156: 'Thomson MO5', 133: 'Odyssey 2 / Videopac G7000', 163: 'SteamVR', 142: 'PC-50X Family', 148: 'AY-3-8607', 146: 'AY-3-8605', 147: 'AY-3-8606', 25: 'Amstrad CPC', 381: 'Playdate', 51: 'Family Computer Disk System', 123: 'WonderSwan Color', 136: 'Neo Geo CD', 35: 'Sega Game Gear', 62: 'Atari Jaguar', 
50: '3DO Interactive Multiplayer', 89: 'Microvision', 150: 'Turbografx-16/PC Engine CD', 23: 'Dreamcast', 65: 'Atari 8-bit', 149: 'PC-9800 Series', 128: 'PC Engine SuperGrafx', 70: 'Vectrex', 85: 'Donner Model 30', 97: 'PDP-8', 98: 'DEC GT40', 112: 'Microcomputer', 101: 'Ferranti Nimrod Computer', 115: 'Apple IIGS', 13: 'DOS', 124: 'SwanCrystal', 127: 'Fairchild Channel F', 87: 'Virtual Boy', 126: 'TRS-80', 130: 'Nintendo Switch', 132: 'Amazon Fire TV', 138: 'VC 4000', 139: '1292 Advanced Programmable Video System', 155: 'Tatung Einstein', 159: 'Nintendo DSi', 119: 'Neo Geo Pocket', 153: 'Dragon 32/64', 154: 
'Amstrad PCW', 11: 'Xbox', 108: 'PDP-11', 53: 'MSX2', 60: 'Atari 7800', 78: 'Sega CD', 24: 'Game Boy Advance', 30: 'Sega 32X', 140: 'AY-3-8500', 143: 'AY-3-8760', 145: 'AY-3-8603', 120: 'Neo Geo Pocket Color', 77: 'Sharp X1', 82: 'Web browser', 109: 'CDC Cyber 70', 113: 'OnLive Game System', 41: 'Wii U', 125: 'PC-8800 Series', 116: 'Acorn Archimedes', 114: 'Amiga CD32', 117: 'Philips CD-i', 121: 'Sharp X68000', 122: 'Nuon', 18: 'Nintendo Entertainment System', 141: 'AY-3-8610', 37: 'Nintendo 3DS', 22: 'Game Boy Color', 64: 'Sega Master System/Mark III', 38: 'PlayStation Portable', 86: 'TurboGrafx-16/PC Engine', 162: 'Oculus VR', 308: 'Playdia', 9: 'PlayStation 3', 14: 'Mac', 306: 'Satellaview', 32: 'Sega Saturn', 15: 'Commodore C64/128/MAX', 66: 'Atari 5200', 
67: 'Intellivision', 73: 'BlackBerry OS', 307: 'Game & Watch', 111: 'Imlac PDS-1', 118: 'FM Towns', 131: 'Nintendo PlayStation', 157: 'NEC PC-6000 Series', 152: 'FM-7', 20: 'Nintendo DS', 63: 'Atari ST/STE', 46: 'PlayStation Vita', 48: 'PlayStation 4', 61: 'Atari Lynx', 93: 'Commodore 16', 21: 'Nintendo GameCube', 42: 'N-Gage', 19: 'Super Nintendo Entertainment System', 374: 'Sharp MZ-2200', 58: 'Super Famicom', 375: 'Epoch Cassette Vision', 388: 'Gear VR', 
96: 'PDP-10', 137: 'New Nintendo 3DS', 377: 'Plug & Play', 57: 'WonderSwan', 71: 'Commodore VIC-20', 75: 'Apple II', 74: 'Windows Phone', 80: 'Neo Geo AES', 84: 'SG-1000', 161: 'Windows Mixed Reality', 79: 'Neo Geo MVS', 33: 'Game Boy', 376: 'Epoch Super Cassette Vision', 5: 'Wii', 382: 'Intellivision Amico', 167: 'PlayStation 5', 387: 'Oculus Go', 385: 'Oculus Rift', 91: 'Bally Astrocade', 384: 'Oculus Quest', 69: 'BBC Microcomputer System', 55: 'Legacy Mobile Device', 379: 'Game.com', 52: 'Arcade', 170: 'Google Stadia', 3: 'Linux', 72: 'Ouya', 95: 'PDP-1', 151: 'TRS-80 Color Computer', 100: 'Analogue electronics', 166: 'PokÃ©mon mini', 102: 'EDSAC', 104: 'HP 2100', 236: 'Exidy Sorcerer', 103: 'PDP-7', 238: 'DVD Player', 105: 'HP 3000', 106: 'SDS Sigma 7', 164: 'Daydream', 107: 'Call-A-Computer time-shared mainframe computer system', 240: 'Zeebo', 110: 'PLATO', 239: 'Blu-ray Player', 26: 'ZX Spectrum', 274: 'PC-FX', 27: 'MSX', 309: 'Evercade', 372: 'OOParts', 373: 'Sinclair ZX81', 203: 'DUPLICATE Stadia', 380: 'Casio Loopy', 169: 'Xbox Series X|S', 12: 'Xbox 360', 49: 'Xbox One', 7: 'PlayStation', 99: 'Family Computer', 389: 'AirConsole', 29: 'Sega Mega Drive/Genesis', 386: 'Meta Quest 2', 390: 'PlayStation VR2', 16: 'Amiga', 47: 'Virtual Console', 165: 'PlayStation VR', 405: 'Windows Mobile', 406: 'Sinclair QL', 407: 'HyperScan', 408: 'Mega Duck/Cougar Boy', 409: 'Legacy Computer', 410: 'Atari Jaguar CD', 411: 'Handheld Electronic LCD', 412: 'Leapster', 413: 'Leapster Explorer/LeadPad Explorer', 414: 'LeapTV', 415: 'Watara/QuickShot Supervision', 416: 'Nintendo 64DD', 417: 'Palm OS', 438: 'Arduboy', 439: 'V.Smile', 441: 'PocketStation', 471: 'Meta Quest 3', 34: 'Android', 59: 'Atari 2600', 68: 'ColecoVision', 440: 'Visual Memory Unit / Visual Memory System', 473: 'Arcadia 2001', 474: 'Gizmondo', 472: 'visionOS', 4: 
'Nintendo 64', 475: 'R-Zone', 476: 'Apple Pippin'}
df = pd.read_csv('andre.csv')


from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    start_delta = datetime.timedelta(days=7)
    last_week = datetime.datetime.now()-start_delta
    print(time.mktime(last_week.timetuple()))
    time_time = time.mktime(last_week.timetuple())

    url = 'https://api.igdb.com/v4/release_dates/'
    headers = {'Client-ID': '4l9k9i1qqdn7ih54tswtrrtr37tdq6', 'Authorization': 'Bearer i1bovfro1q1rfoud62vv8pzlz4map3'}
    myobj = f'fields game.name;where date>{time_time};sort date desc;limit 5;'
    x = requests.post(url,headers=headers,data=myobj)
    list_of_latest = x.json()

    url = 'https://api.igdb.com/v4/games'
    headers = {'Client-ID': '4l9k9i1qqdn7ih54tswtrrtr37tdq6', 'Authorization': 'Bearer i1bovfro1q1rfoud62vv8pzlz4map3'}
    myobj = 'fields name, rating;limit 10;where rating>80;sort rating desc;'

    x = requests.post(url,headers=headers,data=myobj)

    list_of_best = x.json()


    #icons da ps5, pc, xbox 
    # /platform/167
    # /platform/169
    # /platform/6

    return render_template('index.html')

@app.route('/game/<id>')
def get_game(id):
    url = 'https://api.igdb.com/v4/games/'
    headers = {'Client-ID': '4l9k9i1qqdn7ih54tswtrrtr37tdq6', 'Authorization': 'Bearer i1bovfro1q1rfoud62vv8pzlz4map3'}
    myobj = f'fields name, genres, game_modes, platforms,summary,rating; where id = {id};'

    x = requests.post(url,headers=headers,data=myobj)
    i = x.json()[0]

    if ('genres' in i.keys())  and ('game_modes' in i.keys()) and ('platforms' in i.keys()):
        global df
        for gen in i['genres']:
            df[genres_look_up[gen]] = [1]
        for gm in i['game_modes']:
            df[game_modes_look_up[gm]] = [1]
        for plat in i['platforms']:
            df[platforms_look_up[plat]] = [1]
    
        #em df está o input pra barbara
        # ml_output da barbara 
        df = df.drop(['Unnamed: 0.1', 'rating'], axis=1)
        print(df)
        
        with open("model.pkl","rb") as file:
            model = pickle.load(file)
            file.close()
        with open("pca.pkl","rb") as file:
            pca = pickle.load(file)
            file.close()

        df = pca.transform(df)
        ml_output = model.predict(df)

        if ('summary' in i.keys())  and ('name' in i.keys()):
            name = i['name']
            summary = i['summary']
            rating = str(ml_output)
            ai_review = generate_review(name, summary, rating)
            return render_template("gamepage.html", ml_output=ml_output, ai_review=ai_review)

    elif ('summary' in i.keys())  and ('name' in i.keys()) and ('rating' in i.keys()):
        name = i['name']
        summary = i['summary']
        rating = str(i['rating'])
        ai_review = generate_review(name, summary, rating)
        return render_template("gamepage.html", ai_review=ai_review)
    # else
    return render_template("gamepage.html")

@app.route('/platform/<id>')
def get_plat():
    url = 'https://api.igdb.com/v4/games/'
    headers = {'Client-ID': '4l9k9i1qqdn7ih54tswtrrtr37tdq6', 'Authorization': 'Bearer i1bovfro1q1rfoud62vv8pzlz4map3'}
    myobj = f'fields name; where platform = [{id}];limit 50; sort rating desc;'
    x = requests.post(url,headers=headers,data=myobj)
    list_of_games = x.json()



if __name__ == "__main__":
    app.run()