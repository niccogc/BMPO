"""
AIM Authentication Patch

Adds support for custom HTTP headers (like Authorization tokens) to AIM's remote tracking.

Usage:
    import os
    from aim_auth import Run  # Use this instead of: from aim import Run
    
    # Set your auth token
    os.environ['AIM_AUTH_TOKEN'] = 'your-secret-token-here'
    
    # Connect to remote server (token will be automatically added)
    run = Run(repo='aim://your-server.com:53800')
    run.track(0.5, name='loss', step=1)

Cloudflare Access tokens:
    CF_ACCESS_CLIENT_ID: Cloudflare Access Client ID
    CF_ACCESS_CLIENT_SECRET: Cloudflare Access Client Secret
"""

import os
import json
import requests
from aim import Run as OriginalRun
from aim.ext.transport.client import Client

# Store original methods
_original_client_init = Client.__init__
_original_protocol_probe = Client.protocol_probe
_original_connect = Client.connect


def _get_auth_headers():
    """Helper function to get authentication headers from environment variables"""
    headers = {}

    # Option 1: Simple Bearer token
    auth_token = os.getenv('AIM_AUTH_TOKEN')
    if auth_token:
        headers['Authorization'] = f'Bearer {auth_token}'

    # Option 2: Cloudflare Access tokens (takes precedence if both are set)
    cf_client_id = os.getenv('CF_ACCESS_CLIENT_ID')
    cf_client_secret = os.getenv('CF_ACCESS_CLIENT_SECRET')
    if cf_client_id and cf_client_secret:
        headers['CF-Access-Client-Id'] = cf_client_id
        headers['CF-Access-Client-Secret'] = cf_client_secret

    # Option 3: Custom headers from JSON
    custom_headers = os.getenv('AIM_CUSTOM_HEADERS')
    if custom_headers:
        try:
            headers_dict = json.loads(custom_headers)
            headers.update(headers_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse AIM_CUSTOM_HEADERS as JSON: {e}")

    return headers


def _patched_client_init(self, remote_path: str):
    """Patched Client.__init__ that adds authentication headers"""
    # Call original initialization first
    _original_client_init(self, remote_path)

    # Add authentication headers AFTER original initialization
    # This ensures headers are available for all subsequent requests
    try:
        auth_headers = _get_auth_headers()
        if not hasattr(self, 'request_headers'):
            self.request_headers = {}
        self.request_headers.update(auth_headers)
    except ValueError as e:
        raise RuntimeError(f"AIM Auth: {e}")


def _patched_protocol_probe(self):
    """Patched protocol_probe that uses authentication headers"""
    try:
        auth_headers = _get_auth_headers()
    except ValueError as e:
        raise RuntimeError(f"AIM Auth: {e}")

    # Merge with existing request headers
    headers = {**self.request_headers, **auth_headers} if hasattr(self, 'request_headers') else auth_headers

    endpoint = f'http://{self.remote_path}/status/'
    try:
        response = requests.get(endpoint, headers=headers, timeout=10)
        if response.status_code == 200:
            if response.url.startswith('https://'):
                self._http_protocol = 'https://'
                self._ws_protocol = 'wss://'
                return
    except Exception:
        pass

    endpoint = f'https://{self.remote_path}/status/'
    try:
        response = requests.get(endpoint, headers=headers, timeout=10, verify=self.ssl_certfile)
        if response.status_code == 200:
            self._http_protocol = 'https://'
            self._ws_protocol = 'wss://'
    except Exception:
        pass


def _patched_connect(self):
    """Patched connect method that ensures authentication headers are used"""
    # Ensure authentication headers are set
    if not hasattr(self, 'request_headers') or not self.request_headers:
        self.request_headers = {}

    try:
        auth_headers = _get_auth_headers()
        for key, value in auth_headers.items():
            if key not in self.request_headers:
                self.request_headers[key] = value
    except ValueError as e:
        raise RuntimeError(f"AIM Auth: {e}")

    # Call original connect method
    return _original_connect(self)


# Apply the monkey patches
Client.__init__ = _patched_client_init
Client.protocol_probe = _patched_protocol_probe
Client.connect = _patched_connect

# Re-export Run so users can import from this module
Run = OriginalRun

# Export everything AIM normally exports
__all__ = ['Run', 'Client']
