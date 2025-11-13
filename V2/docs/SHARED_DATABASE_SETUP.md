# 🔗 Shared Database Setup

This guide explains how to set up the shared Render PostgreSQL database for team collaboration.

## 🚀 Quick Setup (One-Time)

Run this command once to set up the shared database connection:

```bash
cd V2
python scripts/setup_shared_database.py
```

This copies the database configuration to `~/dot_env.txt` in your home directory. The application will automatically detect and use this file!

## 📋 Manual Setup

If you prefer to set it up manually:

1. **Copy the example file to your home directory:**
   ```bash
   cp V2/dot_env.txt.example ~/dot_env.txt
   ```

2. **That's it!** The application will automatically use this connection.

## ✅ Verify It's Working

After setup, verify the connection:

```bash
cd V2
python scripts/setup_postgres.py
```

You should see:
```
✅ PostgreSQL connection successful!
✅ Database tables created successfully.
```

## 🔄 How It Works

The application checks for database configuration in this priority order:

1. **`DATABASE_URL` environment variable** (highest priority - for overrides)
2. **`~/dot_env.txt` file** (shared team database - **this is what you want!**)
3. **`.env` file** in project directory (project-specific config)
4. **Default SQLite** (fallback for local development)

## 👥 For Your Team

**Everyone on your team should:**

1. Run `python scripts/setup_shared_database.py` (or manually copy `dot_env.txt.example` to `~/dot_env.txt`)
2. Run `python scripts/setup_postgres.py` to initialize the database
3. Start the application: `uvicorn app.main:app --reload`

**Result:** Everyone connects to the same shared database automatically! 🎉

## 🔒 Security Note

The `~/dot_env.txt` file contains database credentials. Make sure:
- ✅ It's in your home directory (not in the project)
- ✅ It's not committed to git (already in `.gitignore`)
- ✅ Only team members have access to it

## 🛠️ Troubleshooting

**Application still using SQLite?**
- Check if `~/dot_env.txt` exists: `ls ~/dot_env.txt`
- Verify the file has the `external db url:` line
- Check file permissions: `chmod 600 ~/dot_env.txt` (optional, for security)

**Connection errors?**
- Run `python scripts/setup_postgres.py` to test the connection
- Verify the database is accessible from your network
- Check firewall settings if connecting from a restricted network

**Want to use a different database?**
- Set `DATABASE_URL` environment variable (overrides home file)
- Or edit `~/dot_env.txt` to point to a different database

## 📝 File Format

The `~/dot_env.txt` file should contain:

```
external db url: postgresql://username:password@host:port/database_name
```

The application automatically extracts the connection string from this line.

---

**Need help?** Check `docs/POSTGRESQL_SETUP.md` for detailed documentation.

