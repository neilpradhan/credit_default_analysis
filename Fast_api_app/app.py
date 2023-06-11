
################################## MAKING  A Flask API APP ###################################

# import json
# # import statsmodels.api as sm
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.metrics import roc_auc_score, roc_curve
# # from imblearn.combine import SMOTEENN 
# import warnings
# warnings.filterwarnings('ignore')
# # from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBClassifier
# # from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
# import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# import numpy as np
# # from sklearn.impute import SimpleImputer
# # from sklearn.preprocessing import MinMaxScaler
# import pickle
# import serverless_wsgi








# # from fastapi import FastAPI
# # from fastapi.responses import JSONResponse
# from flask import Flask
# from flask import jsonify, request
# from flask_marshmallow import Marshmallow
# from flask_cors import CORS, cross_origin
# import requests, json
# from flask import send_from_directory
# # import uvicorn
# # from mangum import Mangum
# # import serverless_wsgi
# # app=FastAPI()


# app = Flask(__name__)
# CORS(app)
# # handler = Mangum(app)



# # @app.route("/hello")
# def hello_world():
#     return "Hello, World!"


# # @app.route("/request",  methods = ['POST'])
# def give_it_back():

#     return request.get_json()





# @app.route("/predict",  methods = ['POST'])
# def testing():
#     '''
#     This function will take the requests and give back the uuid and probability of default as required

#     For empty fields I am assuming that data is just like training data, so the columns which were empty in training data
#     can be filled in test data but some other columns which were non-null are made mandatory. The fields mandatory in this example are

#     'num_arch_ok_0_12m', 'num_unpaid_bills','age', 'num_arch_ok_0_12m'

#     (ofcourse these are amendable based on requirements and design of machine learning solution)

#     INPUT: 

#     {
#     "num_arch_ok_0_12m": [9.0],
#     "num_unpaid_bills": [0.0],
#     "avg_payment_span_0_12m": [null],
#     "age": [25.0],
#     "max_paid_inv_0_24m": [13749.0],
#     "account_status": [1.0],
#     "account_worst_status_0_3m":[null],
#     "account_worst_status_12_24m": [1.0],
#     "account_worst_status_3_6m": [1.0],
#     "account_worst_status_6_12m": [1.0],
#     "uuid": ["1234"]
#     }
    
#     OUTPUT:
#     {"uuid":{"0":"1234"},"default":{"0":0.9846019745}}
    
    
#     '''
#     input_file = request.json
#     input_file = dict(input_file)
#     print("input_file", input_file)


#     model_xgb_pickled = pickle.load(open('xgb_5_cat_5_num.pkl','rb'))


#     # dictionary to Dataframe
#     test_df = pd.DataFrame(input_file)



#     test_uuid_ = test_df['uuid']

#     top_5_categories_with_less_than_5_unique_values = ['account_status',\
#     'account_worst_status_0_3m','account_worst_status_12_24m',\
#     'account_worst_status_3_6m','account_worst_status_6_12m']

#     total_cols_to_consider = ['num_arch_ok_0_12m', 'num_unpaid_bills',\
#                             'avg_payment_span_0_12m', 'age', 'max_paid_inv_0_24m',\
#                             'account_status', 'account_worst_status_0_3m', 'account_worst_status_12_24m',\
#                             'account_worst_status_3_6m', 'account_worst_status_6_12m']
    

#     # for empty fields I am assuming that data like training data, so the NaN fieldS which were empty in training data
#     # can be filled in test data but some other fields should be compulsary

#     d = {'avg_payment_span_0_12m':17.977933,'account_status':1.0,'account_worst_status_0_3m':1.0,
#     'account_worst_status_12_24m':1.0,'account_worst_status_3_6m':1.0,'account_worst_status_6_12m':1.0}

#     # 'num_arch_ok_0_12m', 'num_unpaid_bills','age', 'num_arch_ok_0_12m' should be compulsary (these are amendable based on requirements)

#     for col in total_cols_to_consider:
#         if col in d.keys():
#             test_df[col] = test_df[col].fillna(d[col])

#     test_df = test_df[total_cols_to_consider]

#     X_test = pd.get_dummies(test_df, columns = top_5_categories_with_less_than_5_unique_values)

#     train_dummy_cols = ['num_arch_ok_0_12m', 'num_unpaid_bills', 'avg_payment_span_0_12m',\
#         'age', 'max_paid_inv_0_24m', 'account_status_1.0', 'account_status_2.0',\
#         'account_status_3.0', 'account_status_4.0',\
#         'account_worst_status_0_3m_1.0', 'account_worst_status_0_3m_2.0',\
#         'account_worst_status_0_3m_3.0', 'account_worst_status_0_3m_4.0',\
#         'account_worst_status_12_24m_1.0', 'account_worst_status_12_24m_2.0',\
#         'account_worst_status_12_24m_3.0', 'account_worst_status_12_24m_4.0',\
#         'account_worst_status_3_6m_1.0', 'account_worst_status_3_6m_2.0',\
#         'account_worst_status_3_6m_3.0', 'account_worst_status_3_6m_4.0',\
#         'account_worst_status_6_12m_1.0', 'account_worst_status_6_12m_2.0',\
#         'account_worst_status_6_12m_3.0', 'account_worst_status_6_12m_4.0']


#     X_test = X_test.reindex(columns = train_dummy_cols, fill_value=0)

#     # scale the test data
#     scaler = StandardScaler()
#     X_test_scaled = scaler.fit_transform(X_test)

#     results = model_xgb_pickled.predict_proba(X_test_scaled)[:,1]

#     results_d = pd.DataFrame({'uuid':test_uuid_,'default':results})


#     return results_d.to_json()


# def handler(event, context):
#     return serverless_wsgi.handle_request(app, event, context)


# # if __name__=="__main__":
# #   uvicorn.run(app,host="0.0.0.0",port=9000)

# # if __name__=="__main__":
# #     app.run(host='0.0.0.0', port=80)



################################## MAKING  A FAST API APP ###################################
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import pickle








from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
# import uvicorn
from mangum import Mangum

app=FastAPI()
handler = Mangum(app)




# def hello_world():
#     return "Hello, World!"

@app.post("/getInformation")
async def getInformation(info : Request):
    req_info = await info.json()
    return {
        "status" : "SUCCESS",
        "data" : req_info
    }





@app.post("/predict")
async def testing(info : Request):
        '''
        This function will take the requests and give back the uuid and probability of default as required

        For empty fields I am assuming that data is just like training data, so the columns which were empty in training data
        can be filled in test data but some other columns which were non-null are made mandatory. The fields mandatory in this example are

        'num_arch_ok_0_12m', 'num_unpaid_bills','age', 'num_arch_ok_0_12m'

        (ofcourse these are amendable based on requirements and design of machine learning solution)

        INPUT: 

        {
        "num_arch_ok_0_12m": [9.0],
        "num_unpaid_bills": [0.0],
        "avg_payment_span_0_12m": [null],
        "age": [25.0],
        "max_paid_inv_0_24m": [13749.0],
        "account_status": [1.0],
        "account_worst_status_0_3m":[null],
        "account_worst_status_12_24m": [1.0],
        "account_worst_status_3_6m": [1.0],
        "account_worst_status_6_12m": [1.0],
        "uuid": ["1234"]
        }
        
        OUTPUT:
        {"uuid":{"0":"1234"},"default":{"0":0.9846019745}}
        
        
        '''
        input_file = await info.json()
        input_file = dict(input_file)
    


        model_xgb_pickled = pickle.load(open('xgb_5_cat_5_num.pkl','rb'))


        # dictionary to Dataframe
        test_df = pd.DataFrame(input_file)



        test_uuid_ = test_df['uuid']

        top_5_categories_with_less_than_5_unique_values = ['account_status',\
        'account_worst_status_0_3m','account_worst_status_12_24m',\
        'account_worst_status_3_6m','account_worst_status_6_12m']

        total_cols_to_consider = ['num_arch_ok_0_12m', 'num_unpaid_bills',\
                                'avg_payment_span_0_12m', 'age', 'max_paid_inv_0_24m',\
                                'account_status', 'account_worst_status_0_3m', 'account_worst_status_12_24m',\
                                'account_worst_status_3_6m', 'account_worst_status_6_12m']
        

        # for empty fields I am assuming that data like training data, so the NaN fieldS which were empty in training data
        # can be filled in test data but some other fields should be compulsary

        d = {'avg_payment_span_0_12m':17.977933,'account_status':1.0,'account_worst_status_0_3m':1.0,
        'account_worst_status_12_24m':1.0,'account_worst_status_3_6m':1.0,'account_worst_status_6_12m':1.0}

        # 'num_arch_ok_0_12m', 'num_unpaid_bills','age', 'num_arch_ok_0_12m' should be compulsary (these are amendable based on requirements)

        for col in total_cols_to_consider:
            if col in d.keys():
                test_df[col] = test_df[col].fillna(d[col])

        test_df = test_df[total_cols_to_consider]

        X_test = pd.get_dummies(test_df, columns = top_5_categories_with_less_than_5_unique_values)

        train_dummy_cols = ['num_arch_ok_0_12m', 'num_unpaid_bills', 'avg_payment_span_0_12m',\
            'age', 'max_paid_inv_0_24m', 'account_status_1.0', 'account_status_2.0',\
            'account_status_3.0', 'account_status_4.0',\
            'account_worst_status_0_3m_1.0', 'account_worst_status_0_3m_2.0',\
            'account_worst_status_0_3m_3.0', 'account_worst_status_0_3m_4.0',\
            'account_worst_status_12_24m_1.0', 'account_worst_status_12_24m_2.0',\
            'account_worst_status_12_24m_3.0', 'account_worst_status_12_24m_4.0',\
            'account_worst_status_3_6m_1.0', 'account_worst_status_3_6m_2.0',\
            'account_worst_status_3_6m_3.0', 'account_worst_status_3_6m_4.0',\
            'account_worst_status_6_12m_1.0', 'account_worst_status_6_12m_2.0',\
            'account_worst_status_6_12m_3.0', 'account_worst_status_6_12m_4.0']


        X_test = X_test.reindex(columns = train_dummy_cols, fill_value=0)

        # scale the test data
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)

        results = model_xgb_pickled.predict_proba(X_test_scaled)[:,1]
        print(type(results))

        results_d = pd.DataFrame({'uuid':test_uuid_,'default':results})

        return results_d.to_dict()
         


# if __name__=="__main__":
#   uvicorn.run(app,host="0.0.0.0",port=9000)






