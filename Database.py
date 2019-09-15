from pymongo import MongoClient
client = MongoClient("mongodb+srv://darkemperorVKO:HaloReach666@cluster0-jr8zu.mongodb.net/test?retryWrites=true&w=majority")
db = client.test
collection=db.Attendence_collection_db
#collection.insert_one({'Name':'Vikas'})

def addNewStudent(rollno,st_name,img):
    collection.insert_one({'RollNo':rollno,'Name':str(st_name),'Embedding':img})
def list_all_students():
    records=collection.find()
    return records
def delete_student(rollno):
     collection.delete_one({"RollNo":rollno})
