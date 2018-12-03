
import datasetSQL
import csv

db_path = "database/db.sqlite"
feature_path = "database/features.sqlite"
filecsv = "meta/wavfile.csv"




def initial_database(db_path=db1_path):
    db = datasetSQL.labelSet(db_path)
    db.initialize()
