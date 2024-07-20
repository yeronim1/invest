from pymongo import MongoClient

def initialize_mongo():
    client = MongoClient('localhost', 27017)
    db = client.qa_system
    contexts_collection = db.contexts
    contexts_collection.create_index([("text", "text")])
    print("Text index created successfully.")

if __name__ == "__main__":
    initialize_mongo()