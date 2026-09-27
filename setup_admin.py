import os
import sqlite3
import getpass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "auth", "users.db")


def main():
    if not os.path.exists(DB_PATH):
        print("Database not found:", DB_PATH)
        print("Run the Streamlit app once and create at least one account first.")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "role" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        if "status" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
        if "last_login_at" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN last_login_at TEXT")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS admin_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                target_user_id TEXT,
                details TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.commit()

        users = conn.execute(
            "SELECT user_id, full_name, email, role, status FROM users ORDER BY id"
        ).fetchall()

        if not users:
            print("No accounts found. Create an account in the Streamlit app first.")
            return

        print("\nRegistered accounts:\n")
        for i, (uid, name, email, role, status) in enumerate(users, 1):
            print(f"{i}. {uid} | {name} | {email} | {role} | {status}")

        target = input("\nEnter the User ID to make ADMIN: ").strip().lower()
        row = conn.execute(
            "SELECT user_id FROM users WHERE user_id = ? COLLATE NOCASE",
            (target,),
        ).fetchone()

        if row is None:
            print("User ID not found.")
            return

        conn.execute("UPDATE users SET role = 'admin', status = 'active' WHERE user_id = ?", (row[0],))
        conn.execute(
            "INSERT INTO admin_activity (admin_user_id, action, target_user_id, details) VALUES (?, ?, ?, ?)",
            (row[0], "INITIAL_ADMIN_SETUP", row[0], "Initial administrator created by setup_admin.py"),
        )
        conn.commit()
        print(f"\nSUCCESS: {row[0]} is now an ADMIN account.")
        print("Restart Streamlit and sign in with this account.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
