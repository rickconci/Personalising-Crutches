# 🚀 Quick Start Guide

Get up and running with the shared database in 3 steps!

## Step 1: Set Up Shared Database Connection

```bash
cd V2
python scripts/setup_shared_database.py
```

This copies the database configuration to `~/dot_env.txt`. The application will automatically use this!

## Step 2: Migrate Local Data (Optional)

If you have existing data in a local SQLite database:

```bash
python scripts/migrate_sqlite_to_postgres.py
```

The script will auto-detect your database from `~/dot_env.txt` and migrate all your data.

## Step 3: Run the Application

```bash
./run.sh
```

That's it! The script will:
- ✅ Check database connection
- ✅ Initialize database if needed
- ✅ Start the web server

Then open: **http://localhost:8000**

## What Happens Next?

**All data from the frontend automatically goes to the online database!**

- Creating participants → Saved to PostgreSQL
- Uploading trials → Saved to PostgreSQL
- Survey responses → Saved to PostgreSQL
- Everything is shared with your team in real-time! 🎉

## Troubleshooting

**Database connection fails?**
- Make sure `~/dot_env.txt` exists: `ls ~/dot_env.txt`
- Run: `python scripts/setup_shared_database.py`

**Migration needed?**
- See `MIGRATION_GUIDE.md` for detailed instructions

**Server won't start?**
- Check Python version: `python --version` (needs 3.10+)
- Make sure dependencies are installed
- Check the error message for details

---

For more details, see:
- `SHARED_DATABASE_SETUP.md` - Database setup
- `MIGRATION_GUIDE.md` - Migrating local data
- `docs/POSTGRESQL_SETUP.md` - Full documentation

