# disable_ssl.py
import ssl
import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create a custom SSL context that doesn't verify certificates
class CustomHTTPAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context(ciphers=None)
        # Don't verify SSL certificates
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

# Patch requests to use our custom adapter
session = requests.Session()
session.mount('https://', CustomHTTPAdapter())

# Patch the original requests.get function
original_get = requests.get
def patched_get(*args, **kwargs):
    kwargs['verify'] = False
    return original_get(*args, **kwargs)
requests.get = patched_get

# Also patch requests.post, requests.put, etc. if needed
original_post = requests.post
def patched_post(*args, **kwargs):
    kwargs['verify'] = False
    return original_post(*args, **kwargs)
requests.post = patched_post

# Set this environment variable
import os
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

print("SSL verification disabled for all HTTP requests")