from flask import Flask, render_template, request, url_for, redirect, session
from bson.objectid import ObjectId
from pymongo import MongoClient
import bcrypt
from dotenv import load_dotenv
import os
from pathlib import Path


dotenv_path = Path('../COMPANO_main/emotion-detection/.env')
load_dotenv(dotenv_path=dotenv_path)

device_ID = os.getenv('DEVICE_ID')

#set app as a Flask instance 
app = Flask(__name__)
#encryption relies on secret keys so they could be run
app.secret_key = "testing"


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


@app.route("/",methods=('GET', 'POST'))
def home():
    return render_template("home.html")



@app.route("/register",methods=('GET','POST'))
def register():
    message = ''
    if "email" in session:
        return redirect(url_for('logged_in'))
    if request.method=='POST' and request.form.get('login_email')==None and request.form.get('login_password')==None:
        new_deviceid = request.form.get('device_id')
        new_name = request.form.get('whole_name')
        new_emergency_name = request.form.get("emergency_name")
        new_email = request.form.get('new_email')
        new_phone = request.form.get('phone')
        new_password = request.form.get('new_password')
        new_password1 = request.form.get('new_password1')
        print(request.form.get('login_email'))

        name_found = user_info.find_one({'name':new_name})
        email_found = user_info.find_one({"email": new_email})
        phone_found = user_info.find_one({'phone_number':new_phone})


        if name_found:
            message = 'Someone with this name is already registered.'
            return render_template('register.html',message=message)
        if email_found:
            message = 'This email already exists.'
            return render_template('register.html', message=message)
        if phone_found:
            message = 'An account already exists with this phone number.'
            return render_template('register.html',message=message)
        if new_password != new_password1:
            message = 'Passwords are not matching! Please try again.'
            return render_template('register.html', message=message)
            
        else:
            print("Came in here now")
            hashed = bcrypt.hashpw(new_password1.encode('utf-8'), bcrypt.gensalt())
            user_input = {'device_id':new_deviceid,'name':new_name,'emergency_contact':new_emergency_name,'email': new_email,'phone_number':new_phone,'password': hashed}
            #insert it in the record collection
            user_info.insert_one(user_input)

            user_data = user_info.find_one({"email": new_email})
            user_email = user_data['email']

            #if registered redirect to logged in as the registered user
            logged_in = True
            return render_template('logged_in.html', email=user_email)

    if request.method=='POST' and request.form.get('new_email')==None and request.form.get('new_password')==None:
        message = 'Please login to your account.'
        if "email" in session:
            return redirect(url_for("logged_in"))
        login_email = request.form.get('login_email')
        login_password = request.form.get('login_password')

        email_found = user_info.find_one({"email": login_email})
        if email_found:
            email_val = email_found['email']
            passwordcheck = email_found['password']
            if bcrypt.checkpw(login_password.encode('utf-8'), passwordcheck):
                session["email"] = email_val
                return redirect(url_for('logged_in'))
            else:
                if "email" in session:
                    return redirect(url_for("logged_in"))
                message = 'Wrong password'
                return render_template('register.html', message=message)
        else:
            message = 'Email not found.'
            return render_template('register.html', message=message)
    return render_template('register.html')


@app.route('/logged_in',methods=('GET', 'POST'))
def logged_in():
    if "email" in session:
        email = session["email"]
        associated_name = user_info.find_one({'device_id':device_ID})['name']
        return render_template('logged_in.html', email=email, dashname=associated_name)
    else:
        return redirect(url_for("home"))

@app.route("/logout", methods=["POST", "GET"])
def logout():
    if "email" in session:
        session.pop("email", None)
        return render_template("home.html")
    else:
        return render_template('home.html')



@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')


# @app.route('/create_reminder',methods=['POST'])
# def remind():
#     item = request.form.get("personal_pref")
#     user_preference = {'Preference':item}
#     user_info.insert_one(user_preference)
     # return render_template('dashboard.html')


@app.route('/reminders')
def reminders():
    all_todos = user_preference.find()
    return render_template('reminders.html', todos=all_todos)


@app.route('/add_todo', methods=['POST'])
def add_todo():
    todo_item = request.form.get('add-todo')
    print(todo_item)
    todo_time = request.form.get('add-time')
    print(todo_time)
    user_preference.insert_one({'text': todo_item, 'time': todo_time, 'complete': False})
    return redirect(url_for('reminders'))


@app.route('/complete_todo/<oid>')
def complete_todo(oid):
    #todo_item = user_preference.find_one({'_id': ObjectId(oid)})
    filter = {'_id': ObjectId(oid)}
    #todo_item['complete'] = True
    newvalues = { "$set": { 'complete': True } }
    user_preference.update_one(filter, newvalues)
    return redirect(url_for('reminders'))


@app.route('/delete_completed')
def delete_completed():
    user_preference.delete_many({'complete': True})
    return redirect(url_for('reminders'))


@app.route('/delete_all')
def delete_all():
    user_preference.delete_many({})
    return redirect(url_for('reminders'))




if __name__ == "__main__":
    app.run(debug=True)

