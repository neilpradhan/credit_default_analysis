# Credit Default Analysis

A complete end-to-end machine learning project from model building to deployment in AWS Lambda, built using data coming from real fintech company

[Project report](https://github.com/neilpradhan/credit_default_analysis/blob/main/Credit_Default_Analysis.pdf)

The deployed API url is in the assignment report Chapter 4, and all the other essential details are in the report.

```

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

{
    "uuid": {
        "0": "1234"
    },
    "default": {
        "0": 0.9846019744873047
    }
}

```

* To run the scripts in this project:
 -make a virtual environment and install all dependencies in the [requirements.txt](https://github.com/neilpradhan/credit_default_analysis/blob/main/requirements.txt)

* The [scripts](https://github.com/neilpradhan/credit_default_analysis/tree/main/scripts) folder has all the jupyter notebooks used to produce all the results.




