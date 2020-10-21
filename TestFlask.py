import os
import flask
import pickle
import lightgbm as lgb
import numpy 

app = flask.Flask(__name__)
app.config['DEBUG'] = True

filename = os.path.join(app.static_folder,'final_model.pkl')
bst = pickle.load(open(filename,'rb'))
################################

@app.route('/index1',methods=['GET'])    
def index1(): 
   return flask.render_template('index1.html',args=flask.request.args)

@app.route('/index2',methods=['GET'])    
def index2(): 
   return flask.render_template('index2.html',args=flask.request.args)
   
@app.route('/index3',methods=['GET'])    
def input(): 
   return flask.render_template('index3.html',args=flask.request.args)

@app.route('/index4',methods=['GET','POST'])    
def submit(): 
    
    prediction = flask.request.args.getlist('x')
    prediction = numpy.array(prediction).reshape((1,-1))
    predictionnum=bst.predict(prediction)[0]*100
    
    return flask.render_template('index4.html',predictionnum=predictionnum)#'The probability of default :'+str(bst.predict(prediction)[0]*100) + '%'
################################
app.run()









