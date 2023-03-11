from pymongo import MongoClient
from datetime import datetime

def MongoDB_user():
    client = MongoClient("mongodb+srv://Access1:passedaccess@mongoeval.2h7cybx.mongodb.net/?retryWrites=true&w=majority")
    db = client.get_database('COMPANO')
    user_info = db.userinfo
    return user_info
    
user_info = MongoDB_user()

def MongoDB_preference():
    client = MongoClient("mongodb+srv://Access1:passedaccess@mongoeval.2h7cybx.mongodb.net/?retryWrites=true&w=majority")
    db = client.get_database('COMPANO')
    user_preference = db.userpreference
    return user_preference


user_preference = MongoDB_preference()


overall_preferences = user_preference.find()


time1er = datetime.today().strftime("%H:%M %p")

exact_time = time1er.split(" ")[0]


for preference in overall_preferences:
    print(preference['time'])
    if preference['time'] == exact_time:
        print("Nice")


print(exact_time)
