import sqlite3

DB_PATH = "auth/users.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT user_id, email, full_name, created_at
    FROM users
    ORDER BY created_at DESC
""")

users = cursor.fetchall()

print("\n===== REGISTERED USERS =====\n")

if not users:
    print("No users registered yet.")
else:
    for user in users:
        user_id, email, full_name, created_at = user

        print(f"Name       : {full_name}")
        print(f"User ID    : {user_id}")
        print(f"Email      : {email}")
        print(f"Created At : {created_at}")
        print("-" * 40)

conn.close()