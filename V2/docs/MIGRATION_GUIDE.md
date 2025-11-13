# 🔄 Database Migration Guide

This guide explains how to migrate your local SQLite database to the shared PostgreSQL database.

## Quick Migration

**One-command migration (auto-detects database from ~/dot_env.txt):**

```bash
cd V2
python scripts/migrate_sqlite_to_postgres.py
```

The script will:
1. Auto-detect your PostgreSQL database from `~/dot_env.txt`
2. Ask for confirmation
3. Copy all data (participants, trials, geometries, etc.) to the online database

## Step-by-Step

### 1. Set Up Shared Database Connection

First, make sure you have the shared database configured:

```bash
python scripts/setup_shared_database.py
```

This creates `~/dot_env.txt` with the database connection.

### 2. Run Migration

```bash
python scripts/migrate_sqlite_to_postgres.py
```

**When prompted:**
- SQLite path: Press Enter to use default (`V2/experiments.db`) or enter custom path
- PostgreSQL URL: Press Enter to use the one from `~/dot_env.txt`, or enter custom URL
- Confirm migration: Type `y` to proceed

### 3. Verify Migration

After migration, verify the data:

```bash
python scripts/setup_postgres.py
```

Then check the database or use the API:
```bash
curl http://localhost:8000/api/participants
```

## What Gets Migrated

The migration script copies all data from SQLite to PostgreSQL:

- ✅ **Participants** - All participant records
- ✅ **Crutch Geometries** - All geometry configurations
- ✅ **Trials** - All experimental trials with survey responses
- ✅ **Experiment Sessions** - Session groupings
- ✅ **Data Files** - File upload records
- ✅ **Optimization Runs** - Bayesian optimization history

## Troubleshooting

**Migration fails with "table already exists"?**
- The tables already exist in PostgreSQL
- This is fine - the script will add new records
- Existing records won't be duplicated (based on primary keys)

**Connection errors?**
- Check `~/dot_env.txt` exists and has correct connection string
- Verify network connectivity to the database server
- Test connection: `python scripts/setup_postgres.py --check-only`

**Data not appearing?**
- Check if migration completed successfully
- Verify you're connecting to the correct database
- Check database directly or via API: `curl http://localhost:8000/api/participants`

## After Migration

Once migration is complete:

1. **Verify data is accessible:**
   ```bash
   python scripts/setup_postgres.py
   ```

2. **Start the server:**
   ```bash
   ./run.sh
   # or
   uvicorn app.main:app --reload
   ```

3. **Check the web interface:**
   - Open http://localhost:8000
   - Verify participants and trials appear

4. **All future data** from the frontend will automatically go to the online database!

---

**Note:** After migration, you can keep your local SQLite database as a backup, but all new data will go to PostgreSQL.

