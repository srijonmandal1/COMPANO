from pymongo import MongoClient

def MongoDB():
    client = MongoClient("mongodb+srv://Access1:passedaccess@mongoeval.2h7cybx.mongodb.net/?retryWrites=true&w=majority")
    db = client.get_database('total_user')
    user_info = db.enter
    return user_info
    
user_info = MongoDB()


def MongoDB_preference():
    client = MongoClient("mongodb+srv://Access1:passedaccess@mongoeval.2h7cybx.mongodb.net/?retryWrites=true&w=majority")
    db = client.get_database('COMPANO')
    user_preference = db.userpreference
    return user_preference


user_preference = MongoDB_preference()


user_info.delete_many({})
user_info.delete_many({})

user_preference.delete_many({})
user_preference.delete_many({})