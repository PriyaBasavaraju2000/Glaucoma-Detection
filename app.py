from flask import Flask, render_template, request, redirect, url_for, flash, session
import mysql.connector as mq
from mysql.connector import Error
from markupsafe import Markup
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

MODEL_PATH = 'glucomamodel.h5'
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Train the model first by running transfer_learning_vgg_16.py")

model = load_model(MODEL_PATH)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'

def dbconnection():
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'glucoma'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', 'Radhika@2002'),
        'auth_plugin': os.getenv('DB_AUTH_PLUGIN', 'mysql_native_password')
    }

    try:
        return mq.connect(**db_config)
    except Error as e:
        app.logger.error("DB connect details = %s", db_config)
        app.logger.error(f"Actual DB error: {e}")
        raise RuntimeError(f"Database connection failed: {e}") from e


@app.route('/')
def home():
    return render_template('index.html', title='Login')


@app.route('/userloginpage')
def userloginpage():
    return render_template('userlogin.html', title='Login')

@app.route('/userregisterpage')
def userregisterpage():
    return render_template('userreg.html', title='reg')


@app.route('/uploadimagepage')
def uploadimagepage():
    return render_template('uploadimage.html', title='upload image')


       
@app.route('/userregister', methods=['GET', 'POST'])
def userregister():
    if request.method == 'POST':
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        address = request.form['address']
        try:
            con = dbconnection()
            cursor = con.cursor()
            cursor.execute("select * from user where email=%s", (email,))
            res = cursor.fetchall()
            if not res:
                cursor.execute(
                    "insert into user(fname,lname,email,phone,pass,address) values (%s,%s,%s,%s,%s,%s)",
                    (fname, lname, email, phone, password, address)
                )
                con.commit()
                message = Markup("<h3>Success! User added successfully</h3>")
                flash(message)
                return redirect(url_for('userloginpage'))
            else:
                message = Markup("<h3>Failed! Email Id already exists</h3>")
                flash(message)
                return redirect(url_for('userregisterpage'))
        except Error as e:
            app.logger.error(f"User registration failed: {e}")
            message = Markup("<h3>Database error. Check connection credentials and permissions.</h3>")
            flash(message)
            return redirect(url_for('userregisterpage'))
        finally:
            if 'con' in locals() and con.is_connected():
                con.close()
       

       

            
@app.route('/userlogin', methods=['GET', 'POST'])
def userlogin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            con = dbconnection()
            cursor = con.cursor()
            cursor.execute("select * from user where email=%s and pass=%s", (email, password))
            res = cursor.fetchall()
            if not res:
                message = Markup("<h3>Failed! Invalid Email or Password</h3>")
                flash(message)
                return redirect(url_for('userloginpage'))
            else:
                return render_template('uploadimage.html')
        except Error as e:
            app.logger.error(f"Login failed: {e}")
            flash(Markup("<h3>Database error. Please try again later.</h3>"))
            return redirect(url_for('userloginpage'))
        finally:
            if 'con' in locals() and con.is_connected():
                con.close()
        

def preprocess(image_path):
    test_image = load_img(image_path, target_size = (224,224))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = test_image/255
    return test_image



@app.route('/uploadimage', methods=['GET','POST'])
def uploadimage():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        if not uploaded_file or uploaded_file.filename == '':
            flash(Markup("<h3>Please upload an image file.</h3>"))
            return redirect(url_for('uploadimagepage'))

        os.makedirs('static/uploads', exist_ok=True)
        filename = os.path.join('static', 'uploads', secure_filename(uploaded_file.filename))
        uploaded_file.save(filename)

        dic = {0: 'Not a Glaucoma', 1: 'Glaucoma Defected'}
        test_image = preprocess(filename)
        result = model.predict(test_image)
        result_idx = int(np.argmax(result, axis=1)[0])
        detec = dic.get(result_idx, 'Unknown')

        return render_template('uploadimage.html', result=detec)

    return render_template('uploadimage.html')

if __name__ == '__main__':
    app.run(debug=True)
