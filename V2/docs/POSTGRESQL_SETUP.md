# PostgreSQL Setup Guide

This guide explains how to set up PostgreSQL for the Personalising Crutches application, enabling shared database access for collaborative work.

## 🎯 Why PostgreSQL?

- **Shared Database**: Multiple users can access the same database simultaneously (when using remote/cloud)
- **Better Concurrency**: Handles concurrent reads/writes efficiently
- **Production Ready**: Suitable for production deployments
- **No File Conflicts**: No need to commit database files to git

## ⚠️ Important: Local vs Shared Database

**Local Database (Docker/Local Install):**

- Runs on your computer only
- Each person has their own separate database
- **NOT shared** - data stays on your machine
- Good for: Development, testing, individual work

**Remote/Cloud Database:**

- Runs on a server accessible over the internet
- Everyone connects to the same database
- **Shared** - all users see the same data in real-time
- Good for: Team collaboration, shared experiments, production

**For collaboration, you MUST use a remote/cloud database!**

## 📋 Prerequisites

- Docker and Docker Compose installed (see installation instructions below)
- Python 3.10+ with required packages
- Network access (for remote databases)

### Installing Docker and Docker Compose

**macOS (Recommended):**

1. Download **Docker Desktop for Mac**: <https://www.docker.com/products/docker-desktop/>
   - Choose the version for your Mac (Intel or Apple Silicon)
2. Open the downloaded `.dmg` file
3. Drag Docker to Applications folder
4. Launch Docker Desktop from Applications
5. Wait for Docker to start (whale icon in menu bar should be steady, not animating)
6. Verify installation:

   ```bash
   docker --version
   docker compose version
   ```

**Linux:**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Or use Docker's official installation script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

**Windows:**

1. Download **Docker Desktop for Windows**: <https://www.docker.com/products/docker-desktop/>
2. Run the installer
3. Follow the installation wizard
4. Restart your computer if prompted
5. Launch Docker Desktop
6. Verify installation in PowerShell:

   ```powershell
   docker --version
   docker compose version
   ```

**Note:** Docker Desktop includes both Docker and Docker Compose, so you only need to install one package.

## 🚀 Quick Start

You need a PostgreSQL database running. Choose one of these options:

### Option 1: Docker (Local Development Only - NOT Shared)

**⚠️ Warning: This creates a local database. Other users won't see your data.**

1. **Start PostgreSQL with Docker Compose:**

```bash
cd V2
docker-compose up -d
```

This starts:

- PostgreSQL database on port 5432 (local only)
- pgAdmin (optional database management UI) on port 5050

2. **Create your `.env` file:**

```bash
cp env.example .env
```

The default configuration in `env.example` is already set for the Docker setup:

```env
DATABASE_URL=postgresql://crutches_user:crutches_password@localhost:5432/personalising_crutches
```

3. **Initialize the database:**

```bash
python scripts/setup_postgres.py
```

This script will:

- Check PostgreSQL connection
- Create all database tables
- Create a sample participant (if database is empty)

4. **Start the application:**

```bash
uvicorn app.main:app --reload
```

### Option 2: Local PostgreSQL Installation (Local Development Only - NOT Shared)

**⚠️ Warning: This creates a local database. Other users won't see your data.**

If you prefer to install PostgreSQL directly on your system:

**macOS (using Homebrew):**

```bash
brew install postgresql@15
brew services start postgresql@15
createdb personalising_crutches
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo -u postgres createdb personalising_crutches
```

Then update your `.env`:

```env
DATABASE_URL=postgresql://your_username@localhost:5432/personalising_crutches
```

**Note:** You'll need to create a user and set permissions. See PostgreSQL documentation for details.

### Option 3: Remote PostgreSQL (Cloud/Hosted) - ✅ For Collaboration

**✅ This is what you need for shared data! Everyone connects to the same database.**

Popular cloud PostgreSQL services:

**Free/Cheap Options:**

- **Supabase** (<https://supabase.com>): Free tier, easy setup, includes web UI
- **Neon** (<https://neon.tech>): Serverless PostgreSQL, free tier
- **Railway** (<https://railway.app>): Simple deployment, free tier
- **Render** (<https://render.com>): Free PostgreSQL tier available

**Enterprise Options:**

- **AWS RDS** (<https://aws.amazon.com/rds/>): Full-featured, pay-as-you-go
- **Google Cloud SQL** (<https://cloud.google.com/sql>): GCP managed PostgreSQL
- **Azure Database for PostgreSQL**: Microsoft's managed service
- **Heroku Postgres** (<https://www.heroku.com/postgres>): Simple, addon-based

**Quick Setup Example (Supabase - Recommended for Teams):**

1. Sign up at <https://supabase.com> (free account)
2. Click "New Project"
3. Fill in project details (name, database password, region)
4. Wait for project to be created (~2 minutes)
5. Go to **Settings** → **Database**
6. Find the **Connection string** section
7. Copy the connection string (looks like: `postgresql://postgres.[ref]:[PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres`)
8. Use it in your `.env` file

**For your team:**

- Share the connection string with all team members
- Everyone uses the same `DATABASE_URL` in their `.env` files
- All data is automatically shared!

If you're using a cloud PostgreSQL service (e.g., AWS RDS, Google Cloud SQL, Heroku Postgres, Supabase):

1. **Get your connection string** from your cloud provider
2. **Update `.env` file:**

```env
DATABASE_URL=postgresql://username:password@your-db-host.com:5432/database_name
```

3. **Initialize the database:**

```bash
python scripts/setup_postgres.py
```

## 🔧 Configuration

### Automatic Shared Database Setup

**For team collaboration, use the shared database configuration:**

```bash
python scripts/setup_shared_database.py
```

This copies the shared database connection to `~/dot_env.txt`. The application automatically detects and uses this file!

**Manual setup:**

1. Copy `V2/dot_env.txt.example` to your home directory:

   ```bash
   cp V2/dot_env.txt.example ~/dot_env.txt
   ```

2. The application will automatically use this connection

**Configuration Priority:**

1. `DATABASE_URL` environment variable (highest priority)
2. `~/dot_env.txt` file (shared team database)
3. `.env` file in project directory
4. Default SQLite (fallback)

### Environment Variables

Edit your `.env` file to configure PostgreSQL (alternative to home directory file):

```env
# Database connection
DATABASE_URL=postgresql://username:password@host:port/database_name

# Docker Compose settings (if using local Docker)
POSTGRES_USER=crutches_user
POSTGRES_PASSWORD=crutches_password
POSTGRES_DB=personalising_crutches
POSTGRES_PORT=5432

# pgAdmin settings (optional)
PGADMIN_EMAIL=admin@crutches.local
PGADMIN_PASSWORD=admin
PGADMIN_PORT=5050
```

### Connection String Format

```
postgresql://[username]:[password]@[host]:[port]/[database_name]
```

Examples:

- Local Docker: `postgresql://crutches_user:crutches_password@localhost:5432/personalising_crutches`
- Remote: `postgresql://user:pass@db.example.com:5432/crutches_db`
- With SSL: `postgresql://user:pass@host:5432/db?sslmode=require`

## 📊 Database Management

### Using pgAdmin (Web UI)

If you enabled pgAdmin in docker-compose:

1. Access pgAdmin: <http://localhost:5050>
2. Login with credentials from `.env` (default: `admin@crutches.local` / `admin`)
3. Add server:
   - Host: `postgres` (Docker service name) or `localhost`
   - Port: `5432`
   - Database: `personalising_crutches`
   - Username: `crutches_user`
   - Password: `crutches_password`

### Using Command Line (psql)

```bash
# Connect to local Docker database
docker exec -it personalising_crutches_db psql -U crutches_user -d personalising_crutches

# Or if PostgreSQL is installed locally
psql -h localhost -U crutches_user -d personalising_crutches
```

## 🔄 Migrating from SQLite

If you have existing data in SQLite:

1. **Run the migration script:**

```bash
python scripts/migrate_sqlite_to_postgres.py
```

2. **Follow the prompts** to specify:
   - Path to your SQLite database
   - PostgreSQL connection URL

3. **Update your `.env`** to use PostgreSQL

## 🛠️ Troubleshooting

### Connection Issues

**Problem**: Cannot connect to PostgreSQL

**Solutions**:

1. Check if PostgreSQL is running: `docker ps` (should show `personalising_crutches_db`)
2. Verify connection string in `.env`
3. Check firewall/network settings for remote databases
4. Ensure database exists and user has proper permissions

### Port Already in Use

**Problem**: Port 5432 is already in use

**Solutions**:

1. Change `POSTGRES_PORT` in `.env` and `docker-compose.yml`
2. Update `DATABASE_URL` to use the new port
3. Or stop the conflicting PostgreSQL service

### Permission Denied

**Problem**: User doesn't have permission to create tables

**Solutions**:

1. Ensure user has `CREATE` privileges
2. Check database owner permissions
3. For Docker: verify `POSTGRES_USER` matches connection string

### Migration Errors

**Problem**: Data migration fails

**Solutions**:

1. Ensure PostgreSQL tables don't already exist (or drop them first)
2. Check that SQLite database is not corrupted
3. Verify all required packages are installed: `pip install psycopg2-binary`

## 🔒 Security Best Practices

1. **Change Default Passwords**: Update passwords in `.env` and `docker-compose.yml`
2. **Use Environment Variables**: Never commit `.env` to git (already in `.gitignore`)
3. **SSL for Remote**: Use SSL connections for remote databases
4. **Limit Access**: Restrict database access to necessary IPs/ports
5. **Regular Backups**: Set up automated backups for production databases

## 📝 Switching Back to SQLite

If you need to switch back to SQLite for local development:

1. **Update `.env`:**

```env
DATABASE_URL=sqlite:///./experiments.db
```

2. **Restart the application**

Note: SQLite files are now in `.gitignore` and won't be committed to git.

## 🚀 Production Deployment

For production deployments:

1. **Use managed PostgreSQL service** (AWS RDS, Google Cloud SQL, etc.)
2. **Enable SSL connections**
3. **Set up automated backups**
4. **Use connection pooling** (already configured in `database/connection.py`)
5. **Monitor database performance**
6. **Use strong passwords and rotate them regularly**

## 📚 Additional Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [SQLAlchemy PostgreSQL Guide](https://docs.sqlalchemy.org/en/14/dialects/postgresql.html)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## ✅ Verification

After setup, verify everything works:

1. **Check connection:**

```bash
python scripts/setup_postgres.py
```

2. **Start server:**

```bash
uvicorn app.main:app --reload
```

3. **Access API docs:**
   - <http://localhost:8000/api/docs>

4. **Test database operations:**
   - Create a participant via API
   - Verify it appears in database
   - Check that other users can see it (if using shared database)

---

**Need Help?** Check the troubleshooting section or review the application logs for detailed error messages.
