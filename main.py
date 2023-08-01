from flask import Flask, render_template,request
import pickle

app = Flask(__name__)
# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')

# dump information to that file
clf = pickle.load(file)
file.close()
@app.route("/",methods=["GET", "POST"])
def hello_world():
    if request.method=="POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnynose = int(myDict['runnynose'])
        deffBreath = int(myDict['deffBreath'])

        #code for inference
        inputFeatures = [fever, age, pain, runnynose, deffBreath]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html',inf=round(infProb*100))
    return render_template('index.html')
        
        #return "Hello, World!" + str(infProb)

if __name__ == "__main__":
    app.run(debug=True)