# 🔌 Connecting to Database with DBeaver

This guide shows how to connect to the shared Render PostgreSQL database using DBeaver.

## Connection Details

From your `dot_env.txt` file:

- **Host:** `dpg-d4al5sk9c44c738k6jg0-a.virginia-postgres.render.com`
- **Port:** `5432`
- **Database:** `crutches`
- **Username:** `crutches_user`
- **Password:** `8DcbqkW0GjGX3xgvxcQo3tYf1tOGfThu`

## Step-by-Step Setup

### 1. Open DBeaver and Create New Connection

1. Open DBeaver
2. Click **"New Database Connection"** (plug icon) or go to **Database → New Database Connection**
3. Select **PostgreSQL** from the list
4. Click **Next**

### 2. Enter Connection Details

In the connection settings:

**Main Tab:**

- **Host:** `dpg-d4al5sk9c44c738k6jg0-a.virginia-postgres.render.com`
- **Port:** `5432`
- **Database:** `crutches`
- **Username:** `crutches_user`
- **Password:** `8DcbqkW0GjGX3xgvxcQo3tYf1tOGfThu`
- ✅ Check **"Save password"** (optional, for convenience)

**SSL Tab (Important for Render):**

- ✅ Check **"Use SSL"**
- **SSL Mode:** Select **"require"** or **"prefer"**

### 3. Test Connection

1. Click **"Test Connection"** button at the bottom
2. If this is your first time, DBeaver may ask to download PostgreSQL driver - click **"Download"**
3. Wait for "Connected" message ✅

### 4. Save and Connect

1. Click **"Finish"**
2. Name your connection (e.g., "Personalising Crutches - Render")
3. The connection will appear in your Database Navigator
4. Double-click to connect!

## Quick Connection String Method

Alternatively, you can use the connection string directly:

1. Create new PostgreSQL connection
2. Click **"Connection string"** tab
3. Paste:

   ```
   postgresql://crutches_user:8DcbqkW0GjGX3xgvxcQo3tYf1tOGfThu@dpg-d4al5sk9c44c738k6jg0-a.virginia-postgres.render.com:5432/crutches?sslmode=require
   ```

4. Click **"Test Connection"**

## Troubleshooting

**Connection timeout?**

- Check your internet connection
- Verify the hostname is correct
- Some networks block port 5432 - try using a VPN if needed

**SSL error?**

- Make sure "Use SSL" is checked
- Try SSL Mode: "require" or "verify-full"
- Render databases require SSL

**Authentication failed?**

- Double-check username and password
- Make sure there are no extra spaces
- Verify credentials in your `dot_env.txt` file

**Driver not found?**

- DBeaver will prompt to download PostgreSQL driver
- Click "Download" when prompted
- Or manually: **Window → Preferences → Drivers → PostgreSQL → Download**

## What You'll See

Once connected, you'll see:

- **Schemas** → **public** → **Tables:**
  - `participants`
  - `crutch_geometries`
  - `trials`
  - `experiment_sessions`
  - `data_files`
  - `optimization_runs`

You can now:

- ✅ Browse all your data
- ✅ Run SQL queries
- ✅ Export data
- ✅ View table structures
- ✅ Edit data (be careful!)

## Security Note

⚠️ **Important:** The password is stored in your DBeaver connection settings. Make sure:

- Don't share your DBeaver connection file
- Use DBeaver's password encryption if available
- Only connect from trusted networks

---

**Need help?** Check DBeaver documentation or verify your connection details match `~/dot_env.txt`
