# AIM Remote Tracking with Authentication

## Not on VPN:

1. Import `Run` from `aim_auth` instead of `aim` if not on VPN:
   ```python
   from aim_auth import Run

   run = Run(repo='aim://aimtracking.kosmon.org:443')
   ```

2. Set authentication environment variables:
   - For Cloudflare Access: `CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET`
   - Or Bearer token: `AIM_AUTH_TOKEN`
   - Or custom headers: `AIM_CUSTOM_HEADERS` (JSON string)

## On VPN:

   ```python
   from aim_auth import Run

   run = Run(repo='aim://192.168.5.5:5800')
   ```
