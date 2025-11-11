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

Environment Variables:
    AIM_AUTH_TOKEN: Bearer token for Authorization header
    
Alternative - Cloudflare Access tokens:
    CF_ACCESS_CLIENT_ID: Cloudflare Access Client ID
    CF_ACCESS_CLIENT_SECRET: Cloudflare Access Client Secret
"""

import os
from aim import Run as OriginalRun
from aim.ext.transport.client import Client

# Store original __init__ method
_original_client_init = Client.__init__


def _patched_client_init(self, remote_path: str):
    """Patched Client.__init__ that adds authentication headers"""
    # Call original initialization
    _original_client_init(self, remote_path)
    
    # Add authentication headers if environment variables are set
    
    # Option 1: Simple Bearer token
    auth_token = os.getenv('AIM_AUTH_TOKEN')
    if auth_token:
        self.request_headers['Authorization'] = f'Bearer {auth_token}'
        print(f"[AIM Auth] Added Authorization header to requests")
    
    # Option 2: Cloudflare Access tokens (takes precedence if both are set)
    cf_client_id = os.getenv('CF_ACCESS_CLIENT_ID')
    cf_client_secret = os.getenv('CF_ACCESS_CLIENT_SECRET')
    if cf_client_id and cf_client_secret:
        self.request_headers['CF-Access-Client-Id'] = cf_client_id
        self.request_headers['CF-Access-Client-Secret'] = cf_client_secret
        print(f"[AIM Auth] Added Cloudflare Access headers to requests")
    
    # Option 3: Custom headers from JSON
    custom_headers = os.getenv('AIM_CUSTOM_HEADERS')
    if custom_headers:
        import json
        try:
            headers_dict = json.loads(custom_headers)
            self.request_headers.update(headers_dict)
            print(f"[AIM Auth] Added custom headers: {list(headers_dict.keys())}")
        except json.JSONDecodeError:
            print(f"[AIM Auth] Warning: Failed to parse AIM_CUSTOM_HEADERS as JSON")


# Apply the monkey patch
Client.__init__ = _patched_client_init

# Re-export Run so users can import from this module
Run = OriginalRun

# Export everything AIM normally exports
__all__ = ['Run', 'Client']
