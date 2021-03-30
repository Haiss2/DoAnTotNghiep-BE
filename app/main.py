import json

from typing import List
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware

from . import crud, models
from .database import SessionLocal, engine

# from . import config as cf
from .code import csv_processor as csv
from .code.clustering import Clustering
from .code.make_rule import Rule
from .code.predict import Predict

from .ha_predict import HAPredict

models.Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#
#  Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def list_all( db: Session = Depends(get_db)):
    return db.query(models.Hash).all()

@app.get("/getItem/{key}")
def getItem( key:str, db: Session = Depends(get_db)):
    return crud.get_value(db, key)

@app.delete("/deleteItem/{key}")
def getItem( key:str, db: Session = Depends(get_db)):
    return crud.delete_value(db, key)
    
@app.post("/setItem/{key}")
def setItem( key:str, value: str, db: Session = Depends(get_db)):
    return crud.set_value(db, key, value)


# type_of_data: glass, ecoli, battery
# train_test_full: train, test, full
@app.get("/data/{type_of_data}/{train_test_full}")
def get_data( type_of_data: str, train_test_full: str , db: Session = Depends(get_db)):
    return csv.read_file("app/data/{}_{}_data.csv".format(type_of_data, train_test_full), 'float')

@app.get("/clustering/{type_of_data}")
def clustering( type_of_data: str, db: Session = Depends(get_db)):
    data = csv.read_file("app/data/{}_train_data.csv".format(type_of_data), 'float')
    cf = json.loads(crud.get_value(db, "config_{}".format(type_of_data)))
    clusters  = Clustering(data, int(cf['k_mean'])).clusters
    crud.set_value(db, "{}_clusters".format(type_of_data), str(clusters))
    return clusters

@app.get("/make_rule/{type_of_data}")
def make_rule( type_of_data: str, db: Session = Depends(get_db)):
    data = csv.read_file("app/data/{}_train_data.csv".format(type_of_data), 'float')
    clusters  = json.loads(crud.get_value(db, "{}_clusters".format(type_of_data)))
    cf = json.loads(crud.get_value(db, "config_{}".format(type_of_data)))
    # import pdb; pdb.set_trace()
    rules = Rule(data, clusters, int(cf['num_classes']), int(cf['num_rules'])).colection_rules()
    crud.set_value(db, "{}_rules".format(type_of_data), str(rules))
    return rules

@app.get("/predict/{type_of_data}/{train_test_full}")
def predict( type_of_data: str, train_test_full: str, db: Session = Depends(get_db)):
    data = csv.read_file("app/data/{}_{}_data.csv".format(type_of_data, train_test_full), 'float')
    clusters  = json.loads(crud.get_value(db, "{}_clusters".format(type_of_data)))
    rules  = json.loads(crud.get_value(db, "{}_rules".format(type_of_data)))

    num_properties = {
        "glass": 9,
        "ecoli": 5,
        "battery": 6
    }

    cf = json.loads(crud.get_value(db, "config_{}".format(type_of_data)))
    predictInstant = Predict(num_properties[type_of_data], clusters, rules, int(cf['k_mean']))
    corrects = 0
    raw_predict = []
    edit = []
    editCollection = []
    for ix, record in enumerate(data):
        raw_predict.append(predictInstant.predict(record, rules))
        if int(record[0]) == predictInstant.predict(record, rules)["predict"]:
            corrects += 1
        else:
            pr_rules = predictInstant.predict(record, rules)["rule"]
            cr_rules = predictInstant.get_rule_truth(record)["rule"]
            edit += predictInstant.detect_cluster(pr_rules, cr_rules, record[1: len(record) ])

    for e in edit:
            if 0.5 < e[2] and e[2] < 0.99 and e[2] + e[4] == 1:
                editCollection.append(e)
    def editCollectionSort(x):
        return x[2]

    editCollection.sort(key=editCollectionSort)
    # Done
    result = {
        "corrects": corrects,
        "total": len(data),
        "raw_predict": raw_predict,
    }

    if train_test_full == 'train':
        crud.set_value(db, "{}_edit_fmc_train".format(type_of_data), str(editCollection))
        result["edit"] = editCollection 

    # default fcm_path
    cf = json.loads(crud.get_value(db, "config_{}".format(type_of_data)))
    
    fmc = [[0.5 for i in range(2*int(cf["k_mean"]))] for i in range( num_properties[type_of_data] )]
    crud.set_value(db, "{}_fmc".format(type_of_data), str(fmc))

    return result

@app.get("/ha_predict/{type_of_data}/{train_test_full}")
def ha_predict( type_of_data: str, train_test_full: str, db: Session = Depends(get_db)):
    data = csv.read_file("app/data/{}_{}_data.csv".format(type_of_data, train_test_full), 'float')
    clusters  = json.loads(crud.get_value(db, "{}_clusters".format(type_of_data)))
    rules  = json.loads(crud.get_value(db, "{}_rules".format(type_of_data)))
    fmcs = json.loads(crud.get_value(db, "{}_fmc".format(type_of_data)))
    cf = json.loads(crud.get_value(db, "config_{}".format(type_of_data)))
    num_corrects = HAPredict(data, clusters, rules, fmcs, int(cf["k_mean"])).num_corrects()
    return {
        "corrects": num_corrects,
        "total": len(data)
    }

@app.get("/ha_change_fmc/{type_of_data}")
def ha_change_fmc( type_of_data: str, editRecord:str, old_num_correct: int, db: Session = Depends(get_db)):

    data = csv.read_file("app/data/{}_train_data.csv".format(type_of_data), 'float')
    clusters  = json.loads(crud.get_value(db, "{}_clusters".format(type_of_data)))
    rules  = json.loads(crud.get_value(db, "{}_rules".format(type_of_data)))
    fmcs = json.loads(crud.get_value(db, "{}_fmc".format(type_of_data)))
    
    attr, flase, u_fasle, true, u_true = json.loads(editRecord)

    if true > flase:
        fmcs[attr][2*true] = u_true/u_fasle*fmcs[attr][2*true - 1]
    else:
        fmcs[attr][2*true + 1] = u_true/u_fasle*fmcs[attr][2*true + 2]
    cf = json.loads(crud.get_value(db, "config_{}".format(type_of_data)))
    new_num_corrects = HAPredict(data, clusters, rules, fmcs, int(cf["k_mean"])).num_corrects()
    
    # Done
    result = {
        "new_corrects": new_num_corrects,
        "old_corrects": old_num_correct,
        "total": len(data),
        "editRecord": json.loads(editRecord)
    }
    if old_num_correct <= new_num_corrects:
        result["status"] = "edited"
        crud.set_value(db, "{}_fmc".format(type_of_data), str(fmcs))
    else: 
        result["status"] = "rejected"
        
    return result