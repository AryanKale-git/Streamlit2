import sqlite3
import bcrypt

# Create a database and user table if not exists
def create_users_table():
    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE,
                            email TEXT UNIQUE,
                            password TEXT)''')
        conn.commit()

# Function to register user
def register_user(username, email, password):
    if not email.endswith("@gmail.com"):
        return False

    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()

        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        try:
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                           (username, email, hashed_pw))
            conn.commit()
        except sqlite3.IntegrityError:
            return False  # Username or email already exists

    return True

# Function to verify login
def login_user(username, email, password):
    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ? AND email = ?", (username, email))
        user = cursor.fetchone()

    return user and bcrypt.checkpw(password.encode('utf-8'), user[0])

# Function to reset password
def reset_password(email, new_password):
    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()

        if user:
            hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            cursor.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_pw, email))
            conn.commit()
            return True

    return False

# Initialize database
create_users_table()
