import requests
from urllib.request import urlopen
import json
import datetime
import time

start_delta = datetime.timedelta(days=7)
last_week = datetime.datetime.now()-start_delta
print(time.mktime(last_week.timetuple()))
time_time = time.mktime(last_week.timetuple())

url = 'https://api.igdb.com/v4/release_dates/'
headers = {'Client-ID': '4l9k9i1qqdn7ih54tswtrrtr37tdq6', 'Authorization': 'Bearer i1bovfro1q1rfoud62vv8pzlz4map3'}
myobj = f'fields game.rating,date, game.name;where date>{time_time};sort date desc;limit 10;'

x = requests.post(url,headers=headers,data=myobj)

print(x.json())

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run()