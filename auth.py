from functools import wraps
from flask import request
import os

def get_endpoints(env_var_name):
    val = os.environ.get(env_var_name)
    if val:
         return [e.strip() for e in val.split(",")]
    return []

valid_tokens = {
    os.environ.get('TOKEN_1'): {
        "holder": os.environ.get('HOLDER_1'),
        "endpoints": get_endpoints('USER_ENDPOINTS')
    },
    os.environ.get('TOKEN_2'): {
        "holder": os.environ.get('HOLDER_2'),
        "endpoints": get_endpoints('DEV_ENDPOINTS')
    }
}


def is_valid_token(token):
    if token not in valid_tokens:
        return False
    return True
    
def get_token_from_request():
    try:
        header = request.headers.get('Authorization')
        if header is not None:
            split_header = header.split(' ')
            if len(split_header) != 2:
                return None
            token = split_header[1]
            return token
    except Exception as e:
        return None
    
def require_token(original_function):
    @wraps(original_function)
    def wrapper_function(*args, **kwargs):
        token = get_token_from_request()
        if is_valid_token(token):
            return original_function(*args, **kwargs)
        return{
            "Status": "Unauthorized",    
        }
    return wrapper_function
