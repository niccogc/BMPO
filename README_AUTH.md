# AIM Authentication Setup

This project includes `aim_auth.py` which adds authentication support to AIM's remote tracking.

## Quick Start

### 1. Basic Usage

```python
import os
from aim_auth import Run  # Use aim_auth instead of aim

# Set your token
os.environ['AIM_AUTH_TOKEN'] = 'your-secret-token'

# Connect to remote server (will automatically add Authorization header)
run = Run(repo='aim://your-server.com:53800')
run.track(0.5, name='loss', step=1)
```

### 2. Run the Example

```bash
# Edit example_with_auth.py to set your server URL and token
python example_with_auth.py
```

## Authentication Methods

### Option 1: Bearer Token (Simple)

Best for: Simple token authentication, Cloudflare Tunnel with custom auth

```python
os.environ['AIM_AUTH_TOKEN'] = 'your-secret-token'
```

This adds the header: `Authorization: Bearer your-secret-token`

### Option 2: Cloudflare Access Service Tokens

Best for: Cloudflare Access protected endpoints

```python
os.environ['CF_ACCESS_CLIENT_ID'] = 'your-client-id.access'
os.environ['CF_ACCESS_CLIENT_SECRET'] = 'your-secret-key'
```

This adds the headers:
- `CF-Access-Client-Id: your-client-id.access`
- `CF-Access-Client-Secret: your-secret-key`

### Option 3: Custom Headers (Advanced)

Best for: Any custom authentication scheme

```python
import os
import json

headers = {
    "X-API-Key": "your-api-key",
    "X-Custom-Header": "custom-value"
}
os.environ['AIM_CUSTOM_HEADERS'] = json.dumps(headers)
```

## Cloudflare Setup Guide

### Step 1: Create Cloudflare Tunnel

```bash
# Install cloudflared
# On your server, create a tunnel to your AIM server
cloudflared tunnel create aim-tracking

# Route traffic
cloudflared tunnel route dns aim-tracking aim.yourdomain.com

# Run tunnel (in your AIM server)
cloudflared tunnel run --url http://localhost:53800 aim-tracking
```

### Step 2: Add Cloudflare Access (or custom auth)

**Option A: Cloudflare Access (Recommended)**

1. Go to Cloudflare Zero Trust Dashboard
2. Create an Application for `aim.yourdomain.com`
3. Create a Service Token:
   - Go to Access > Service Auth > Service Tokens
   - Click "Create Service Token"
   - Save the Client ID and Client Secret
4. Use in your code:
   ```python
   os.environ['CF_ACCESS_CLIENT_ID'] = 'the-client-id.access'
   os.environ['CF_ACCESS_CLIENT_SECRET'] = 'the-secret'
   ```

**Option B: Custom Worker (Simple Bearer Token)**

Create a Cloudflare Worker:

```javascript
export default {
  async fetch(request, env) {
    const authHeader = request.headers.get('Authorization');
    const expectedToken = 'Bearer your-secret-token';
    
    if (authHeader !== expectedToken) {
      return new Response('Unauthorized', { status: 401 });
    }
    
    // Forward to origin
    return fetch(request);
  }
}
```

Then use:
```python
os.environ['AIM_AUTH_TOKEN'] = 'your-secret-token'
```

### Step 3: Update Your Training Code

```python
# Old code:
# from aim import Run

# New code:
from aim_auth import Run  # Just change this one line!

# Everything else stays the same
run = Run(repo='aim://aim.yourdomain.com:443')  # Use port 443 for HTTPS
run.track(loss, name='loss', step=step)
```

## How It Works

The `aim_auth.py` script uses monkey patching to intercept AIM's `Client` initialization and inject custom headers into all HTTP requests. This is completely transparent to your training code.

## Security Best Practices

1. **Never commit tokens to git**
   - Use environment variables
   - Use `.env` files (add to .gitignore)
   - Use secret management systems

2. **Use HTTPS in production**
   ```python
   run = Run(repo='aim://your-server.com:443')  # Port 443 for HTTPS
   ```

3. **Rotate tokens regularly**
   - Cloudflare Access tokens can be regenerated
   - Update environment variables across your training servers

4. **Use different tokens per environment**
   - Development token
   - Production token
   - Per-team tokens if needed

## Troubleshooting

### "Failed to connect to Aim Server"

1. Check if server is running: `curl https://aim.yourdomain.com/status/`
2. Check if token is correct
3. Check Cloudflare Access logs

### "Unauthorized" / 401 errors

1. Verify token is set: `echo $AIM_AUTH_TOKEN`
2. Check token format (should not include "Bearer" prefix in env var)
3. Verify Cloudflare Access token is still valid

### Headers not being sent

Make sure you import from `aim_auth`, not `aim`:
```python
from aim_auth import Run  # Correct
# from aim import Run     # Wrong - won't have auth
```

## Advanced: Using with Remote Servers

When running training on remote servers (AWS, GCP, etc.):

```bash
# Set environment variable on the remote server
export AIM_AUTH_TOKEN='your-secret-token'

# Or pass it when running your script
AIM_AUTH_TOKEN='your-token' python train.py

# Or use a .env file
echo "AIM_AUTH_TOKEN=your-token" > .env
python train.py  # Make sure to load .env in your code
```

## Example: Production Setup

```python
import os
from pathlib import Path
from aim_auth import Run

# Load token from file (better than hardcoding)
token_file = Path.home() / '.aim_token'
if token_file.exists():
    os.environ['AIM_AUTH_TOKEN'] = token_file.read_text().strip()
else:
    raise RuntimeError("Token file not found. Create ~/.aim_token with your token")

# Connect to production server
run = Run(repo='aim://aim.yourdomain.com:443')

# Your training code...
```
