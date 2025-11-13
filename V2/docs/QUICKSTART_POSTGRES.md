# 🚀 Quick Start: PostgreSQL Setup

Get PostgreSQL running in 3 steps!

## ⚠️ Important: Local vs Shared Database

**For Development (Local):**

- Docker or local PostgreSQL = Each person has their own database
- Good for: Testing, development, individual work
- **NOT shared** - data stays on your machine

**For Collaboration (Shared):**

- Remote/cloud PostgreSQL = Everyone connects to the same database
- Good for: Team collaboration, shared experiments
- **Shared** - all users see the same data in real-time

## Step 1: Set Up PostgreSQL Database

Choose based on your needs:

### Option A: Docker (Local Development Only - NOT Shared)

**⚠️ This creates a local database on your machine. Other users won't see your data.**

**Install Docker Desktop:**

- **macOS**: Download from <https://www.docker.com/products/docker-desktop/>
- **Linux**: `sudo apt-get install docker.io docker-compose` or use Docker's install script
- **Windows**: Download Docker Desktop from the same link

**Start PostgreSQL:**

```bash
cd V2
docker-compose up -d
```

### Option B: Local PostgreSQL Installation (Local Development Only - NOT Shared)

**⚠️ This creates a local database on your machine. Other users won't see your data.**

**macOS:**

```bash
brew install postgresql@15
brew services start postgresql@15
createdb personalising_crutches
```

**Linux:**

```bash
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo -u postgres createdb personalising_crutches
```

### Option C: Remote/Cloud Database (✅ For Collaboration & Sharing)

**✅ This is what you need for shared data! Everyone connects to the same database.**

Popular options:

- **Supabase** (Free tier): <https://supabase.com> - Easiest to set up
- **Neon** (Free tier): <https://neon.tech> - Serverless PostgreSQL
- **AWS RDS**: <https://aws.amazon.com/rds/> - Enterprise-grade
- **Google Cloud SQL**: <https://cloud.google.com/sql> - GCP option
- **Heroku Postgres**: <https://www.heroku.com/postgres> - Simple hosting
- **Railway**: <https://railway.app> - Easy deployment

**Quick Setup Example (Supabase):**

1. Sign up at <https://supabase.com>
2. Create a new project
3. Go to Settings → Database
4. Copy the connection string (looks like: `postgresql://postgres:[PASSWORD]@[HOST]:5432/postgres`)
5. Use it in your `.env` file

## Step 2: Configure Environment

### Option 1: Shared Database (Recommended for Teams)

**Automatic setup using home directory file:**

```bash
python scripts/setup_shared_database.py
```

This copies the shared database configuration to `~/dot_env.txt`. The application will automatically use this database connection!

**Manual setup:**

1. Copy `dot_env.txt.example` to your home directory:

   ```bash
   cp V2/dot_env.txt.example ~/dot_env.txt
   ```

2. The application will automatically detect and use this file

### Option 2: Project-specific Configuration

If you prefer to use a project-specific `.env` file:

```bash
cp env.example .env
```

**Edit `.env` and set your `DATABASE_URL`:**

- **Docker (Local)**: Already configured in `env.example` (no changes needed)
- **Local PostgreSQL**: `postgresql://your_username@localhost:5432/personalising_crutches`
- **Remote/Cloud (Shared)**: `postgresql://username:password@host:port/database_name`

**Configuration Priority:**

1. `DATABASE_URL` environment variable (highest priority)
2. `~/dot_env.txt` file (shared database)
3. `.env` file in project directory
4. Default SQLite (fallback)

## Step 3: Initialize Database

```bash
python scripts/setup_postgres.py
```

This script connects to your PostgreSQL database (wherever it is) and sets up the schema.

Done! 🎉

## Start the Application

```bash
uvicorn app.main:app --reload
```

Access:

- Web Interface: <http://localhost:8000>
- API Docs: <http://localhost:8000/api/docs>
- pgAdmin: <http://localhost:5050> (optional)

## Migrating from SQLite?

If you have existing SQLite data:

```bash
python scripts/migrate_sqlite_to_postgres.py
```

## Need More Details?

See the full guide: [docs/POSTGRESQL_SETUP.md](docs/POSTGRESQL_SETUP.md)

## Troubleshooting

**Can't connect to PostgreSQL?**

- **Docker**: Make sure Docker Desktop is running (`docker ps`)
- **Local**: Check PostgreSQL is running (`brew services list` on macOS, `sudo systemctl status postgresql` on Linux)
- **Remote**: Verify network connectivity and credentials
- Verify `.env` has correct `DATABASE_URL`
- Wait a few seconds after starting (database needs time to initialize)

**Port already in use?**

- Change `POSTGRES_PORT` in `.env` and `docker-compose.yml`
- Update `DATABASE_URL` to match

**Permission denied?**

- On Linux, add your user to docker group: `sudo usermod -aG docker $USER`
- Log out and back in, or restart your terminal
