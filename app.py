import os
import re
import sqlite3
import hashlib
import hmac
import secrets
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import predict_student_performance
from src.utils import (
    get_performance_category,
    get_risk_level,
    get_recommendations,
)

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AcademicAI | Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR, "dataset", "student_performance_dataset1_corrected.csv"
)
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "student_performance_model.pkl"
)


# Local authentication database.
# Add auth/users.db to .gitignore before pushing the project.
AUTH_DIR = os.path.join(BASE_DIR, "auth")
AUTH_DB_PATH = os.path.join(AUTH_DIR, "users.db")
os.makedirs(AUTH_DIR, exist_ok=True)

# =========================================================
# MODERN FRONTEND
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #f6f8fc;
    --surface: #ffffff;
    --ink: #101828;
    --muted: #667085;
    --line: #e4e7ec;
    --primary: #635bff;
    --primary-dark: #4f46e5;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 82% 0%, rgba(99,91,255,.12), transparent 28%),
        radial-gradient(circle at 4% 18%, rgba(6,182,212,.08), transparent 24%),
        var(--bg);
}

[data-testid="stHeader"] { background: rgba(246,248,252,.78); }

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1020 0%, #171d35 100%);
    border-right: 1px solid rgba(255,255,255,.07);
}
section[data-testid="stSidebar"] * { color: #eef2ff !important; }
section[data-testid="stSidebar"] .stRadio label {
    border-radius: 12px;
    padding: 8px 10px;
    margin: 2px 0;
}

.stButton > button {
    min-height: 44px;
    border-radius: 12px;
    border: 1px solid #dfe3ea;
    font-weight: 700;
    transition: all .18s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    border-color: #b9b4ff;
    box-shadow: 0 8px 20px rgba(99,91,255,.12);
}
.primary-btn button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white;
    border: none;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,.88);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 25px rgba(16,24,40,.05);
}
div[data-testid="stMetricLabel"] { color: var(--muted); }
div[data-testid="stMetricValue"] { color: var(--ink); font-weight: 800; }

/* Authentication */
/* Use a real Streamlit keyed container instead of a raw HTML wrapper.
   This prevents the large blank vertical area seen on the login page. */
.st-key-auth_page {
    width: min(1120px, 100%);
    margin: 0 auto !important;
    padding: 0 !important;
}
.st-key-auth_page > div:first-child {
    align-items: stretch !important;
}
.st-key-auth_page [data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
}
.st-key-auth_page [data-testid="column"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
.st-key-auth_form_panel {
    min-height: 610px;
    height: 100%;
    padding: 42px 48px !important;
    background: #fff;
    border-radius: 0 24px 24px 0;
    box-sizing: border-box;
}
.st-key-auth_form_panel > div:first-child {
    padding: 0 !important;
}
.auth-brand-panel {
    min-height: 610px;
    height: 100%;
    padding: 42px 48px;
    color: white;
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 12% 15%, rgba(255,255,255,.18), transparent 23%),
        radial-gradient(circle at 88% 88%, rgba(6,182,212,.18), transparent 25%),
        linear-gradient(145deg, #0b1020 0%, #27235f 52%, #635bff 100%);
}
.auth-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 20px;
    font-weight: 800;
}
.auth-logo-mark {
    width: 44px; height: 44px;
    display: grid; place-items: center;
    border-radius: 14px;
    background: rgba(255,255,255,.13);
    border: 1px solid rgba(255,255,255,.17);
    font-size: 23px;
}
.auth-kicker {
    margin-top: 58px;
    color: #c7d2fe;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 800;
}
.auth-title {
    font-size: clamp(38px, 4.2vw, 58px);
    line-height: 1.04;
    letter-spacing: -2.5px;
    font-weight: 800;
    margin: 13px 0 20px;
}
.auth-description {
    max-width: 510px;
    color: #dbe4ff;
    font-size: 16px;
    line-height: 1.8;
}
.auth-benefit {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-top: 18px;
    color: #f4f6ff;
    font-size: 14px;
}
.auth-check {
    width: 25px; height: 25px;
    display: grid; place-items: center;
    border-radius: 50%;
    background: rgba(255,255,255,.12);
    color: #a5f3fc;
}
.auth-form-panel {
    min-height: 610px;
    height: 100%;
    padding: 42px 48px;
    background: #fff;
    border-radius: 0;
}
.auth-form-panel .stTextInput label { font-weight: 700; color: #344054; }
.auth-form-panel input { border-radius: 12px !important; }
.auth-heading {
    color: var(--ink);
    font-size: 31px;
    font-weight: 800;
    letter-spacing: -.8px;
    margin: 7px 0;
}
.auth-subheading {
    color: var(--muted);
    line-height: 1.65;
    margin-bottom: 22px;
}
.auth-mode {
    color: var(--muted);
    font-size: 13px;
    margin-top: 17px;
    text-align: center;
}
.auth-security {
    color: #667085;
    font-size: 12px;
    margin-top: 18px;
    padding: 11px 13px;
    border-radius: 11px;
    background: #f8fafc;
    border: 1px solid #eef2f6;
}
.auth-password-note {
    font-size: 12px;
    color: #98a2b3;
    margin-top: -8px;
    margin-bottom: 12px;
}

/* Admin */
.admin-hero {
    background: linear-gradient(135deg, #0f172a 0%, #312e81 55%, #635bff 100%);
    color: white; border-radius: 24px; padding: 30px;
    box-shadow: 0 18px 45px rgba(49,46,129,.18); margin-bottom: 22px;
}
.admin-kicker { color:#c7d2fe; font-size:12px; font-weight:800; letter-spacing:1.8px; text-transform:uppercase; }
.admin-title { font-size:36px; font-weight:800; margin:6px 0; letter-spacing:-1px; }
.admin-sub { color:#dbe4ff; line-height:1.6; }
.admin-card { background:#fff; border:1px solid #e4e7ec; border-radius:18px; padding:20px; box-shadow:0 8px 25px rgba(16,24,40,.05); }
.status-active { color:#067647; background:#ecfdf3; border:1px solid #abefc6; padding:4px 9px; border-radius:999px; font-weight:700; font-size:12px; }
.status-disabled { color:#b42318; background:#fef3f2; border:1px solid #fecdca; padding:4px 9px; border-radius:999px; font-weight:700; font-size:12px; }
.role-admin { color:#6941c6; background:#f4f3ff; border:1px solid #d9d6fe; padding:4px 9px; border-radius:999px; font-weight:700; font-size:12px; }
.role-user { color:#344054; background:#f2f4f7; border:1px solid #d0d5dd; padding:4px 9px; border-radius:999px; font-weight:700; font-size:12px; }

/* Main application */
.hero {
    background:
        radial-gradient(circle at 90% 10%, rgba(99,91,255,.18), transparent 28%),
        linear-gradient(135deg, #fff, #f7f7ff);
    border: 1px solid #e7e8f4;
    border-radius: 28px;
    padding: 44px;
    box-shadow: 0 15px 45px rgba(16,24,40,.06);
}
.eyebrow {
    color: var(--primary);
    font-size: 13px;
    letter-spacing: 1.8px;
    font-weight: 800;
    text-transform: uppercase;
}
.hero-title {
    color: var(--ink);
    font-size: clamp(38px, 5vw, 68px);
    line-height: 1.03;
    font-weight: 800;
    letter-spacing: -2.5px;
    margin: 12px 0 18px;
}
.hero-sub {
    color: #475467;
    font-size: 18px;
    line-height: 1.7;
    max-width: 760px;
}
.glass-card {
    background: rgba(255,255,255,.88);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(16,24,40,.045);
    height: 100%;
}
.feature-card { min-height: 190px; }
.icon-box {
    width: 48px; height: 48px;
    display: grid; place-items: center;
    border-radius: 14px;
    background: #eef0ff;
    font-size: 23px;
}
.section-title { font-size: 26px; color: var(--ink); font-weight: 800; }
.page-head { margin-bottom: 25px; }
.page-kicker {
    color: var(--primary);
    font-weight: 700;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.page-title { color: var(--ink); font-size: 38px; font-weight: 800; margin: 4px 0; }
.muted { color: var(--muted); }
.result-card {
    background: linear-gradient(135deg, #111827, #312e81);
    color: white;
    border-radius: 24px;
    padding: 30px;
    box-shadow: 0 18px 45px rgba(49,46,129,.18);
}
.result-number { font-size: 54px; font-weight: 800; line-height: 1; }
.badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,.13);
    border: 1px solid rgba(255,255,255,.16);
    font-size: 13px;
}
.footer {
    margin-top: 50px;
    padding: 22px;
    text-align: center;
    color: #98a2b3;
    border-top: 1px solid var(--line);
}
.small-note { font-size: 12px; color: #98a2b3; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# AUTHENTICATION
# =========================================================
def get_db_connection():
    connection = sqlite3.connect(AUTH_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_auth_database():
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_login_at TEXT
            )
            """
        )

        # Upgrade older users.db files without deleting existing accounts.
        columns = {row[1] for row in connection.execute("PRAGMA table_info(users)").fetchall()}
        if "role" not in columns:
            connection.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        if "status" not in columns:
            connection.execute("ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
        if "last_login_at" not in columns:
            connection.execute("ALTER TABLE users ADD COLUMN last_login_at TEXT")

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                target_user_id TEXT,
                details TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def normalize_user_id(value):
    return value.strip().lower()


def normalize_email(value):
    return value.strip().lower()


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)

    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        120_000,
    )
    return derived_key.hex(), salt


def verify_password(password, stored_hash, salt):
    calculated_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(calculated_hash, stored_hash)


def valid_email(email):
    return bool(
        re.fullmatch(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            email,
        )
    )


def valid_user_id(user_id):
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]{4,24}", user_id))


def password_is_strong(password):
    return bool(
        len(password) >= 8
        and re.search(r"[A-Z]", password)
        and re.search(r"[a-z]", password)
        and re.search(r"\d", password)
    )


def get_user(identifier):
    identifier = identifier.strip()
    with get_db_connection() as connection:
        return connection.execute(
            """
            SELECT *
            FROM users
            WHERE user_id = ? COLLATE NOCASE
               OR email = ? COLLATE NOCASE
            LIMIT 1
            """,
            (identifier, identifier),
        ).fetchone()


def create_user(user_id, email, full_name, password):
    user_id = normalize_user_id(user_id)
    email = normalize_email(email)
    full_name = full_name.strip()
    password_hash_value, salt = hash_password(password)

    try:
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO users
                (user_id, email, full_name, password_hash, salt)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, email, full_name, password_hash_value, salt),
            )
            connection.commit()
        return True, ""
    except sqlite3.IntegrityError as exc:
        message = str(exc).lower()
        if "user_id" in message:
            return False, "This User ID already exists. Please choose another."
        if "email" in message:
            return False, "This email address is already registered."
        return False, "An account with these details already exists."


def authenticate_user(identifier, password):
    user = get_user(identifier)
    if user is None:
        return None

    if user["status"] != "active":
        return None

    if verify_password(password, user["password_hash"], user["salt"]):
        with get_db_connection() as connection:
            connection.execute(
                "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?",
                (user["id"],),
            )
            connection.commit()
        return get_user(identifier)

    return None


def is_admin():
    return st.session_state.get("role", "user") == "admin"


def admin_log(action, target_user_id=None, details=""):
    admin_id = st.session_state.get("username", "system")
    with get_db_connection() as connection:
        connection.execute(
            "INSERT INTO admin_activity (admin_user_id, action, target_user_id, details) VALUES (?, ?, ?, ?)",
            (admin_id, action, target_user_id, details),
        )
        connection.commit()


def get_all_users(search=""):
    with get_db_connection() as connection:
        if search.strip():
            q = f"%{search.strip()}%"
            return connection.execute(
                """SELECT id, user_id, full_name, email, role, status, created_at, last_login_at
                   FROM users
                   WHERE user_id LIKE ? COLLATE NOCASE
                      OR email LIKE ? COLLATE NOCASE
                      OR full_name LIKE ? COLLATE NOCASE
                   ORDER BY id DESC""",
                (q, q, q),
            ).fetchall()
        return connection.execute(
            """SELECT id, user_id, full_name, email, role, status, created_at, last_login_at
               FROM users ORDER BY id DESC"""
        ).fetchall()


def count_admins():
    with get_db_connection() as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM users WHERE role = 'admin' AND status = 'active'"
        ).fetchone()[0]


def update_user_role(user_id, role):
    with get_db_connection() as connection:
        connection.execute("UPDATE users SET role = ? WHERE user_id = ?", (role, user_id))
        connection.commit()


def update_user_status(user_id, status):
    with get_db_connection() as connection:
        connection.execute("UPDATE users SET status = ? WHERE user_id = ?", (status, user_id))
        connection.commit()


def delete_user(user_id):
    with get_db_connection() as connection:
        connection.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        connection.commit()


def reset_user_password(user_id, new_password):
    password_hash_value, salt = hash_password(new_password)
    with get_db_connection() as connection:
        connection.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE user_id = ?",
            (password_hash_value, salt, user_id),
        )
        connection.commit()


def _secret_value(name, default=""):
    """Read a Streamlit secret without exposing it in the application UI."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        return default
    if value is None:
        return default
    return str(value).strip()


def bootstrap_admin_from_secrets():
    """Create or repair the first admin account on Streamlit Cloud.

    Required Streamlit secrets:
        ADMIN_USER_ID
        ADMIN_EMAIL
        ADMIN_FULL_NAME
        ADMIN_PASSWORD

    Optional:
        ADMIN_FORCE_PASSWORD_RESET = "true"

    The password is never stored in source code or GitHub. Only its salted
    PBKDF2 hash is stored in the local SQLite database.
    """
    admin_user_id = normalize_user_id(_secret_value("ADMIN_USER_ID"))
    admin_email = normalize_email(_secret_value("ADMIN_EMAIL"))
    admin_full_name = _secret_value("ADMIN_FULL_NAME")
    admin_password = _secret_value("ADMIN_PASSWORD")
    force_reset = _secret_value("ADMIN_FORCE_PASSWORD_RESET", "false").lower() in {
        "1", "true", "yes", "on"
    }

    # If deployment secrets are not configured, do nothing.
    if not all([admin_user_id, admin_email, admin_full_name, admin_password]):
        return

    if not valid_user_id(admin_user_id) or not valid_email(admin_email):
        return

    if not password_is_strong(admin_password):
        return

    password_hash_value, salt = hash_password(admin_password)

    with get_db_connection() as connection:
        existing = connection.execute(
            "SELECT id, user_id, email FROM users WHERE user_id = ? COLLATE NOCASE",
            (admin_user_id,),
        ).fetchone()

        if existing is None:
            # If the requested User ID is new but the email already exists,
            # update that existing account instead of creating a duplicate.
            existing_by_email = connection.execute(
                "SELECT id, user_id FROM users WHERE email = ? COLLATE NOCASE",
                (admin_email,),
            ).fetchone()

            if existing_by_email is None:
                connection.execute(
                    """
                    INSERT INTO users
                    (user_id, email, full_name, password_hash, salt, role, status)
                    VALUES (?, ?, ?, ?, ?, 'admin', 'active')
                    """,
                    (
                        admin_user_id,
                        admin_email,
                        admin_full_name,
                        password_hash_value,
                        salt,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO admin_activity
                    (admin_user_id, action, target_user_id, details)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        admin_user_id,
                        "BOOTSTRAP_ADMIN_CREATED",
                        admin_user_id,
                        "Administrator created from Streamlit Secrets",
                    ),
                )
            else:
                target_id = existing_by_email["user_id"]
                connection.execute(
                    """
                    UPDATE users
                    SET user_id = ?, full_name = ?, role = 'admin', status = 'active'
                    WHERE id = ?
                    """,
                    (admin_user_id, admin_full_name, existing_by_email["id"]),
                )
                if force_reset:
                    connection.execute(
                        "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
                        (password_hash_value, salt, existing_by_email["id"]),
                    )
                connection.execute(
                    """
                    INSERT INTO admin_activity
                    (admin_user_id, action, target_user_id, details)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        admin_user_id,
                        "BOOTSTRAP_ADMIN_REPAIRED",
                        admin_user_id,
                        f"Administrator repaired from Streamlit Secrets; previous user ID: {target_id}",
                    ),
                )
        else:
            connection.execute(
                """
                UPDATE users
                SET email = ?, full_name = ?, role = 'admin', status = 'active'
                WHERE id = ?
                """,
                (admin_email, admin_full_name, existing["id"]),
            )
            if force_reset:
                connection.execute(
                    "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
                    (password_hash_value, salt, existing["id"]),
                )

        connection.commit()


def login_screen():
    """Render the authentication screen without the old blank vertical wrapper."""
    with st.container(key="auth_page"):
        left, right = st.columns([1.05, 0.95], gap="small")

        with left:
            st.markdown(
                """
                <div class="auth-brand-panel">
                    <div class="auth-logo">
                        <div class="auth-logo-mark">🎓</div>
                        AcademicAI
                    </div>
                    <div class="auth-kicker">SMART ACADEMIC ANALYTICS</div>
                    <div class="auth-title">
                        Your academic data.<br>
                        Smarter decisions.
                    </div>
                    <div class="auth-description">
                        A professional Machine Learning workspace for predicting
                        student performance, exploring academic data and planning
                        measurable improvement goals.
                    </div>
                    <div class="auth-benefit">
                        <div class="auth-check">✓</div>
                        <div>AI-powered performance prediction</div>
                    </div>
                    <div class="auth-benefit">
                        <div class="auth-check">✓</div>
                        <div>Interactive academic analytics</div>
                    </div>
                    <div class="auth-benefit">
                        <div class="auth-check">✓</div>
                        <div>Model insights and explainability</div>
                    </div>
                    <div class="auth-benefit">
                        <div class="auth-check">✓</div>
                        <div>Goal planning and recommendations</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            with st.container(key="auth_form_panel"):
                if "auth_mode" not in st.session_state:
                    st.session_state.auth_mode = "signin"

                c1, c2 = st.columns(2)
                with c1:
                    if st.button(
                        "Sign In",
                        use_container_width=True,
                        type="primary" if st.session_state.auth_mode == "signin" else "secondary",
                        key="auth_tab_signin",
                    ):
                        st.session_state.auth_mode = "signin"
                        st.rerun()

                with c2:
                    if st.button(
                        "Create Account",
                        use_container_width=True,
                        type="primary" if st.session_state.auth_mode == "register" else "secondary",
                        key="auth_tab_register",
                    ):
                        st.session_state.auth_mode = "register"
                        st.rerun()

                st.write("")

                if st.session_state.auth_mode == "signin":
                    st.markdown(
                        """
                        <div class="auth-kicker" style="margin-top:0;color:#635bff;">
                            WELCOME BACK
                        </div>
                        <div class="auth-heading">Sign in to AcademicAI</div>
                        <div class="auth-subheading">
                            Use your registered User ID or email address to access
                            your analytics workspace.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    identifier = st.text_input(
                        "User ID or Email Address",
                        placeholder="Enter your User ID or email",
                        key="login_identifier",
                    )
                    password = st.text_input(
                        "Password",
                        type="password",
                        placeholder="Enter your password",
                        key="login_password",
                    )

                    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
                    login_clicked = st.button(
                        "🔐  Sign In",
                        use_container_width=True,
                        type="primary",
                        key="login_submit",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    if login_clicked:
                        if not identifier.strip() or not password:
                            st.error("Please enter both your User ID/email and password.")
                        else:
                            user = authenticate_user(identifier, password)
                            if user is not None:
                                st.session_state.authenticated = True
                                st.session_state.username = user["user_id"]
                                st.session_state.full_name = user["full_name"]
                                st.session_state.user_email = user["email"]
                                st.session_state.role = user["role"]
                                st.session_state.status = user["status"]
                                st.session_state.page = "🏠 Dashboard"
                                st.rerun()
                            else:
                                st.error(
                                    "Account not found, the password is incorrect, "
                                    "or this account is disabled."
                                )

                    st.markdown(
                        """
                        <div class="auth-security">
                            🔒 Passwords are stored as salted PBKDF2 hashes rather than plain text.
                        </div>
                        <div class="auth-mode">
                            Don't have an account? Click <b>Create Account</b> above.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:
                    st.markdown(
                        """
                        <div class="auth-kicker" style="margin-top:0;color:#635bff;">
                            NEW ACCOUNT
                        </div>
                        <div class="auth-heading">Create your account</div>
                        <div class="auth-subheading">
                            Register once, then sign in using your User ID or email address.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    full_name = st.text_input(
                        "Full Name",
                        placeholder="Enter your full name",
                        key="register_name",
                    )
                    user_id = st.text_input(
                        "User ID",
                        placeholder="e.g. prathmesh_ai",
                        key="register_user_id",
                    )
                    email = st.text_input(
                        "Email Address",
                        placeholder="you@example.com",
                        key="register_email",
                    )
                    password = st.text_input(
                        "Create Password",
                        type="password",
                        placeholder="Minimum 8 characters",
                        key="register_password",
                    )
                    confirm_password = st.text_input(
                        "Confirm Password",
                        type="password",
                        placeholder="Re-enter your password",
                        key="register_confirm_password",
                    )

                    st.markdown(
                        """
                        <div class="auth-password-note">
                            Password: 8+ characters, one uppercase letter, one lowercase letter and one number.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
                    register_clicked = st.button(
                        "✨  Create Account",
                        use_container_width=True,
                        type="primary",
                        key="register_submit",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    if register_clicked:
                        user_id_clean = normalize_user_id(user_id)
                        email_clean = normalize_email(email)
                        full_name_clean = full_name.strip()

                        if not full_name_clean:
                            st.error("Please enter your full name.")
                        elif not valid_user_id(user_id_clean):
                            st.error(
                                "User ID must be 4–24 characters and can contain "
                                "letters, numbers, dots, hyphens or underscores."
                            )
                        elif not valid_email(email_clean):
                            st.error("Please enter a valid email address.")
                        elif not password_is_strong(password):
                            st.error(
                                "Password must contain 8+ characters, uppercase, lowercase and a number."
                            )
                        elif password != confirm_password:
                            st.error("Passwords do not match.")
                        else:
                            success, message = create_user(
                                user_id_clean,
                                email_clean,
                                full_name_clean,
                                password,
                            )
                            if success:
                                st.session_state.auth_mode = "signin"
                                st.session_state.login_identifier = user_id_clean
                                st.session_state.login_password = ""
                                st.success("Account created successfully. You can now sign in.")
                                st.rerun()
                            else:
                                st.error(message)

                    st.markdown(
                        """
                        <div class="auth-mode">
                            Already have an account? Click <b>Sign In</b> above.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


initialize_auth_database()
bootstrap_admin_from_secrets()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_screen()
    st.stop()

# =========================================================
# LOAD DATA & MODEL
# =========================================================
@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)


try:
    df = load_dataset(DATASET_PATH)
    model = load_pipeline(MODEL_PATH)
except Exception as exc:
    st.error("The application could not load the dataset or trained model.")
    st.exception(exc)
    st.stop()

# =========================================================
# SESSION STATE
# =========================================================
defaults = {
    "predicted_marks": None,
    "performance": None,
    "risk": None,
    "recommendations": [],
    "page": "🏠 Dashboard",
    "full_name": "",
    "user_email": "",
    "role": "user",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:10px 5px 20px;">
            <div style="font-size:26px;">🎓</div>
            <div style="font-size:20px;font-weight:800;">AcademicAI</div>
            <div style="font-size:12px;color:#98a2b3 !important;">
                Student Performance Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_admin():
        # Admin workspace intentionally contains only operational pages.
        # Analytics-only pages remain available to normal users.
        pages = [
            "🏠 Dashboard",
            "📊 Student Prediction",
            "🎯 Goal Planner",
            "👑 Admin Control Center",
            "ℹ️ About",
        ]
    else:
        pages = [
            "🏠 Dashboard",
            "📊 Student Prediction",
            "📈 Dataset Insights",
            "🧠 Feature Importance",
            "🤖 Model Performance",
            "🎯 Goal Planner",
            "ℹ️ About",
        ]

    # Never leave an admin on an analytics-only page after a role change,
    # and never allow a normal user to retain an admin page in session state.
    if st.session_state.page not in pages:
        st.session_state.page = "🏠 Dashboard"

    selected_page = st.radio(
        "WORKSPACE",
        pages,
        index=pages.index(st.session_state.page)
        if st.session_state.page in pages
        else 0,
    )
    st.session_state.page = selected_page

    st.markdown("---")

    st.markdown(
        f"""
        <div style="
            background:rgba(255,255,255,.07);
            border:1px solid rgba(255,255,255,.08);
            padding:15px;
            border-radius:16px;
        ">
            <div style="font-size:11px;color:#98a2b3 !important;
                        letter-spacing:1.2px;font-weight:800;">
                ACCOUNT
            </div>
            <div style="font-weight:800;margin-top:6px;">
                {st.session_state.get("full_name", "User")}
            </div>
            <div style="font-size:12px;color:#98a2b3 !important;margin-top:3px;">
                @{st.session_state.get("username", "user")}
            </div>
            <div style="font-size:11px;color:#98a2b3 !important;margin-top:2px;">
                {st.session_state.get("user_email", "")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.full_name = ""
        st.session_state.user_email = ""
        st.session_state.role = "user"
        st.session_state.status = ""
        st.session_state.page = "🏠 Dashboard"
        st.rerun()

# =========================================================
# COMMON HELPERS
# =========================================================
def page_header(kicker, title, description):
    st.markdown(
        f"""
        <div class="page-head">
            <div class="page-kicker">{kicker}</div>
            <div class="page-title">{title}</div>
            <div class="muted">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_student_data(
    gender,
    study_hours,
    attendance,
    previous_marks,
    assignment,
    quiz,
    internal,
    internet,
    extra,
    sleep,
):
    return pd.DataFrame(
        {
            "Gender": [gender],
            "Study_Hours": [study_hours],
            "Attendance_Percent": [attendance],
            "Previous_Marks": [previous_marks],
            "Assignment_Score": [assignment],
            "Quiz_Score": [quiz],
            "Internal_Marks": [internal],
            "Internet_Access": [internet],
            "Extracurricular": [extra],
            "Sleep_Hours": [sleep],
        }
    )


def run_prediction(student_data):
    predicted = predict_student_performance(student_data)
    if isinstance(predicted, (np.ndarray, list)):
        value = float(predicted[0])
    else:
        value = float(predicted)

    value = max(0.0, min(100.0, value))
    return round(value, 2)


# =========================================================
# ROLE-BASED PAGE ACCESS
# =========================================================
ADMIN_PAGES = {
    "🏠 Dashboard",
    "📊 Student Prediction",
    "🎯 Goal Planner",
    "👑 Admin Control Center",
    "ℹ️ About",
}
USER_PAGES = {
    "🏠 Dashboard",
    "📊 Student Prediction",
    "📈 Dataset Insights",
    "🧠 Feature Importance",
    "🤖 Model Performance",
    "🎯 Goal Planner",
    "ℹ️ About",
}

allowed_pages = ADMIN_PAGES if is_admin() else USER_PAGES
if st.session_state.page not in allowed_pages:
    st.session_state.page = "🏠 Dashboard"


# =========================================================
# DASHBOARD
# =========================================================
if st.session_state.page == "🏠 Dashboard":
    first_name = st.session_state.get("full_name", "Student").split()[0]

    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">AI-POWERED ACADEMIC ANALYTICS</div>
            <div class="hero-title">
                Welcome, {first_name}.<br>
                Student Performance<br>
                Prediction & Analytics
            </div>
            <div class="hero-sub">
                Transform student data into predictions, insights and
                actionable academic recommendations through Machine Learning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    b1, b2, b3 = st.columns(3)

    with b1:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("📊 Start Prediction", use_container_width=True):
            st.session_state.page = "📊 Student Prediction"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with b2:
        if st.button("📈 Explore Dataset", use_container_width=True):
            st.session_state.page = "📈 Dataset Insights"
            st.rerun()

    with b3:
        if st.button("🎯 Open Goal Planner", use_container_width=True):
            st.session_state.page = "🎯 Goal Planner"
            st.rerun()

    st.write("")

    # Live project statistics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Student Records", f"{len(df):,}")
    c2.metric("Total Features", df.shape[1])
    c3.metric("Predictor Features", 10)
    c4.metric("Target", "Final Marks")

    st.write("")
    st.markdown('<div class="section-title">What you can do</div>', unsafe_allow_html=True)
    st.caption("A unified workspace for prediction, analytics and academic planning.")

    cards = [
        ("🎯", "Performance Prediction", "Estimate final marks from student academic inputs using the trained ML pipeline."),
        ("📊", "Data Analytics", "Explore distributions, relationships, statistics and student performance patterns."),
        ("🧠", "Feature Analysis", "Understand model coefficients and which processed features have stronger influence."),
        ("🤖", "Model Evaluation", "Review R², MAE, MSE, RMSE and actual-versus-predicted performance."),
        ("📅", "Goal Planning", "Set a target and generate practical improvement suggestions."),
        ("💡", "Academic Insights", "Convert model output into performance category, risk level and recommendations."),
    ]

    for start in range(0, len(cards), 3):
        row = st.columns(3)
        for col, item in zip(row, cards[start:start + 3]):
            icon, title, text = item
            with col:
                st.markdown(
                    f"""
                    <div class="glass-card feature-card">
                        <div class="icon-box">{icon}</div>
                        <h3 style="color:#111827;margin-bottom:8px;">{title}</h3>
                        <p class="muted" style="line-height:1.65;">{text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.write("")
    st.markdown('<div class="section-title">How the system works</div>', unsafe_allow_html=True)
    st.caption("End-to-end workflow from student input to academic insight.")

    workflow = [
        ("01", "Student Data", "Academic and learning-related inputs"),
        ("02", "Preprocessing", "Scaling and categorical encoding"),
        ("03", "ML Pipeline", "Trained Linear Regression model"),
        ("04", "Prediction", "Estimated final marks"),
        ("05", "Insights", "Performance, risk and recommendations"),
    ]

    row = st.columns(5)
    for col, item in zip(row, workflow):
        num, title, text = item
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center;">
                    <div style="font-size:12px;font-weight:800;color:#635bff;">{num}</div>
                    <h4 style="color:#111827;">{title}</h4>
                    <div class="muted" style="font-size:13px;line-height:1.5;">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =========================================================
# STUDENT PREDICTION
# =========================================================
elif st.session_state.page == "📊 Student Prediction":
    page_header(
        "PREDICTION ENGINE",
        "Student Performance Prediction",
        "Enter student information and generate an estimated final marks prediction.",
    )

    left, right = st.columns([1.45, .75], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("### 📝 Student Profile")

            a, b = st.columns(2)
            with a:
                gender = st.selectbox("Gender", ["Male", "Female"])
                study_hours = st.slider("Study Hours / Day", 0.0, 12.0, 6.0, 0.5)
                attendance = st.slider("Attendance (%)", 50, 100, 80)
                previous_marks = st.slider("Previous Marks", 30, 95, 70)
                assignment = st.slider("Assignment Score", 40, 100, 75)

            with b:
                quiz = st.slider("Quiz Score", 30, 100, 70)
                internal = st.slider("Internal Marks", 10, 30, 22)
                internet = st.selectbox("Internet Access", ["Yes", "No"])
                extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
                sleep = st.slider("Sleep Hours", 4.0, 10.0, 7.0, 0.5)

            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            predict_btn = st.button(
                "🚀 Generate AI Prediction",
                use_container_width=True,
                type="primary",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if predict_btn:
                student_data = build_student_data(
                    gender,
                    study_hours,
                    attendance,
                    previous_marks,
                    assignment,
                    quiz,
                    internal,
                    internet,
                    extra,
                    sleep,
                )

                try:
                    value = run_prediction(student_data)
                    st.session_state.predicted_marks = value
                    st.session_state.performance = get_performance_category(value)
                    st.session_state.risk = get_risk_level(value)
                    st.session_state.recommendations = get_recommendations(value)
                except Exception as exc:
                    st.error("Prediction failed.")
                    st.exception(exc)

    with right:
        st.markdown(
            """
            <div class="glass-card">
                <div class="icon-box">🤖</div>
                <h3 style="color:#111827;">Prediction engine</h3>
                <p class="muted" style="line-height:1.7;">
                    The application sends the entered values through the
                    saved Scikit-Learn preprocessing and Machine Learning
                    pipeline.
                </p>
                <hr>
                <div class="muted">10 predictor features</div>
                <div class="muted">Linear Regression pipeline</div>
                <div class="muted">Output constrained to 0–100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.predicted_marks is not None:
        st.write("")
        st.markdown(
            f"""
            <div class="result-card">
                <div class="badge">AI PREDICTION RESULT</div>
                <div style="margin-top:20px;color:#c7d2fe;">Predicted Final Marks</div>
                <div class="result-number">{st.session_state.predicted_marks:.2f}</div>
                <div style="margin-top:18px;">
                    <span class="badge">{st.session_state.performance}</span>
                    &nbsp;
                    <span class="badge">Risk: {st.session_state.risk}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("### 💡 Personalized Recommendations")
        if st.session_state.recommendations:
            for rec in st.session_state.recommendations:
                st.success(rec)
        else:
            st.info("No additional recommendations were generated.")

# =========================================================
# DATASET INSIGHTS
# =========================================================
elif st.session_state.page == "📈 Dataset Insights":
    page_header(
        "DATA EXPLORATION",
        "Dataset Insights",
        "Explore the 5,000-student dataset through statistics and interactive visualizations.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Records", f"{df.shape[0]:,}")
    c2.metric("📑 Features", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.write("")
    with st.expander("📄 Dataset Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    with st.expander("📋 Dataset Information"):
        info = pd.DataFrame(
            {
                "Column Name": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Missing Values": df.isnull().sum().values,
            }
        )
        st.dataframe(info, use_container_width=True, hide_index=True)

    with st.expander("📊 Statistical Summary"):
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown("### 📊 Exploratory Data Analysis")

    charts = [
        ("👨‍🎓 Gender Distribution", px.bar(
            df["Gender"].value_counts().reset_index(),
            x="Gender", y="count", text="count",
            title="Gender Distribution",
        )),
        ("📊 Attendance Distribution", px.histogram(
            df, x="Attendance_Percent", nbins=20,
            title="Attendance Percentage Distribution",
        )),
        ("📚 Study Hours Distribution", px.histogram(
            df, x="Study_Hours", nbins=15,
            title="Study Hours Distribution",
        )),
        ("📝 Previous Marks Distribution", px.histogram(
            df, x="Previous_Marks", nbins=20,
            title="Previous Marks Distribution",
        )),
        ("🎯 Final Marks Distribution", px.histogram(
            df, x="Final_Marks", nbins=20,
            title="Final Marks Distribution",
        )),
        ("😴 Sleep Hours Distribution", px.histogram(
            df, x="Sleep_Hours", nbins=15,
            title="Sleep Hours Distribution",
        )),
        ("🌐 Internet Access", px.bar(
            df["Internet_Access"].value_counts().reset_index(),
            x="Internet_Access", y="count", text="count",
            title="Internet Access Distribution",
        )),
        ("🏃 Extracurricular Activities", px.bar(
            df["Extracurricular"].value_counts().reset_index(),
            x="Extracurricular", y="count", text="count",
            title="Extracurricular Activities Distribution",
        )),
    ]

    for start in range(0, len(charts), 2):
        row = st.columns(2)
        for col, (title, fig) in zip(row, charts[start:start + 2]):
            with col:
                st.markdown(f"**{title}**")
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 Feature Relationships with Final Marks")

    relationships = [
        ("Attendance vs Final Marks", "Attendance_Percent"),
        ("Study Hours vs Final Marks", "Study_Hours"),
        ("Previous Marks vs Final Marks", "Previous_Marks"),
        ("Assignment Score vs Final Marks", "Assignment_Score"),
        ("Quiz Score vs Final Marks", "Quiz_Score"),
        ("Internal Marks vs Final Marks", "Internal_Marks"),
    ]

    for start in range(0, len(relationships), 2):
        row = st.columns(2)
        for col, (title, xcol) in zip(row, relationships[start:start + 2]):
            with col:
                fig = px.scatter(
                    df,
                    x=xcol,
                    y="Final_Marks",
                    color="Gender",
                    title=title,
                    opacity=.65,
                )
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================
elif st.session_state.page == "🧠 Feature Importance":
    page_header(
        "MODEL INTERPRETABILITY",
        "Feature Importance",
        "Analyze the trained Linear Regression coefficients and their relative magnitude.",
    )

    try:
        lr_model = model.named_steps["model"]
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        importance = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": lr_model.coef_,
            }
        )
        importance["Absolute"] = importance["Coefficient"].abs()
        importance = importance.sort_values("Absolute", ascending=False)

        c1, c2 = st.columns(2)
        c1.metric("Processed Features", len(feature_names))
        c2.metric("Coefficients", len(lr_model.coef_))

        fig = px.bar(
            importance.sort_values("Absolute"),
            x="Absolute",
            y="Feature",
            orientation="h",
            title="Absolute Coefficient Magnitude",
            labels={"Absolute": "Absolute coefficient"},
        )
        fig.update_layout(template="plotly_white", height=650)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Coefficient Details")
        st.dataframe(
            importance[["Feature", "Coefficient", "Absolute"]],
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:
        st.error("Could not extract feature importance from the saved pipeline.")
        st.exception(exc)

# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif st.session_state.page == "🤖 Model Performance":
    page_header(
        "MODEL EVALUATION",
        "Model Performance",
        "Evaluate the saved Linear Regression pipeline against the available dataset.",
    )

    from sklearn.metrics import (
        r2_score,
        mean_absolute_error,
        mean_squared_error,
    )

    X = df.drop(columns=["Final_Marks"])
    y_true = df["Final_Marks"]
    y_pred = model.predict(X)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score", f"{r2:.4f}")
    m2.metric("MAE", f"{mae:.2f}")
    m3.metric("MSE", f"{mse:.2f}")
    m4.metric("RMSE", f"{rmse:.2f}")

    eval_df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    eval_df["Residual"] = eval_df["Actual"] - eval_df["Predicted"]

    c1, c2 = st.columns(2)

    with c1:
        fig = px.scatter(
            eval_df,
            x="Actual",
            y="Predicted",
            opacity=.55,
            title="Actual vs Predicted Final Marks",
        )
        min_val = min(eval_df["Actual"].min(), eval_df["Predicted"].min())
        max_val = max(eval_df["Actual"].max(), eval_df["Predicted"].max())
        fig.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(dash="dash", width=2),
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            eval_df,
            x="Residual",
            nbins=30,
            title="Residual Distribution",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("⚙️ View Pipeline Architecture"):
        st.code(str(model), language="text")

# =========================================================
# GOAL PLANNER
# =========================================================
elif st.session_state.page == "🎯 Goal Planner":
    page_header(
        "ACADEMIC PLANNING",
        "Goal Planner",
        "Set a target and receive improvement suggestions based on the current prediction.",
    )

    left, right = st.columns(2)

    with left:
        gender = st.selectbox("Gender", ["Male", "Female"], key="goal_gender")
        study_hours = st.slider(
            "Study Hours",
            0.0, 12.0, 6.0, .5,
            key="goal_study",
        )
        attendance = st.slider(
            "Attendance %",
            50, 100, 80,
            key="goal_att",
        )
        previous_marks = st.slider(
            "Previous Marks",
            30, 95, 70,
            key="goal_prev",
        )
        assignment = st.slider(
            "Assignment Score",
            40, 100, 75,
            key="goal_assign",
        )

    with right:
        quiz = st.slider(
            "Quiz Score",
            30, 100, 70,
            key="goal_quiz",
        )
        internal = st.slider(
            "Internal Marks",
            10, 30, 22,
            key="goal_internal",
        )
        internet = st.selectbox(
            "Internet Access",
            ["Yes", "No"],
            key="goal_net",
        )
        extra = st.selectbox(
            "Extracurricular",
            ["Yes", "No"],
            key="goal_extra",
        )
        sleep = st.slider(
            "Sleep Hours",
            4.0, 10.0, 7.0, .5,
            key="goal_sleep",
        )

    target = st.slider("🎯 Target Marks", 40, 100, 85)

    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    plan_btn = st.button(
        "✨ Generate Goal Plan",
        use_container_width=True,
        type="primary",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if plan_btn:
        student = build_student_data(
            gender,
            study_hours,
            attendance,
            previous_marks,
            assignment,
            quiz,
            internal,
            internet,
            extra,
            sleep,
        )

        try:
            current = float(model.predict(student)[0])
            current = max(0, min(100, current))

            c1, c2 = st.columns(2)
            c1.metric("Current Predicted Marks", f"{current:.2f}")
            c2.metric("Target Marks", target)

            if current >= target:
                st.success("🎉 Congratulations! Your current prediction meets the target.")
            else:
                gap = target - current
                st.warning(f"You need approximately {gap:.1f} more marks to reach the target.")

                improvements = []

                if study_hours < 8:
                    improvements.append(
                        f"📚 Increase Study Hours from {study_hours:.1f} → {min(8, study_hours + 2):.1f} hrs/day."
                    )
                if attendance < 95:
                    improvements.append(
                        f"🏫 Improve Attendance from {attendance}% → {min(95, attendance + 10)}%."
                    )
                if assignment < 95:
                    improvements.append(
                        f"📝 Improve Assignment Score from {assignment} → {min(95, assignment + 10)}."
                    )
                if quiz < 95:
                    improvements.append(
                        f"🧪 Improve Quiz Score from {quiz} → {min(95, quiz + 10)}."
                    )
                if internal < 28:
                    improvements.append(
                        f"📖 Improve Internal Marks from {internal} → {min(28, internal + 4)}."
                    )
                if sleep < 7 or sleep > 8.5:
                    improvements.append("😴 Maintain a consistent 7–8 hours of sleep.")
                if internet == "No":
                    improvements.append("🌐 Use online learning resources whenever possible.")
                if extra == "Yes":
                    improvements.append("⚖ Balance extracurricular activities with study time.")

                st.markdown("### Recommended Improvements")
                for item in improvements:
                    st.info(item)

                estimated = current
                estimated += min(2, max(0, 8 - study_hours)) * 2
                estimated += min(5, (95 - attendance) / 10)
                estimated += min(3, (95 - assignment) / 10)
                estimated += min(3, (95 - quiz) / 10)
                estimated += min(2, (28 - internal) / 4)
                estimated = min(100, estimated)

                st.metric("Estimated Achievable Score", f"{estimated:.2f}")

                if estimated >= target:
                    st.success("🎯 The target looks achievable with these improvements.")
                else:
                    st.info("💡 The target is ambitious. Continue improving consistently.")

        except Exception as exc:
            st.error("Goal plan generation failed.")
            st.exception(exc)

# =========================================================
# ADMIN CONTROL CENTER
# =========================================================
if st.session_state.page == "👑 Admin Control Center":
    if not is_admin():
        st.error("You do not have permission to access the Admin Control Center.")
        st.session_state.page = "🏠 Dashboard"
        st.rerun()

    page_header(
        "ADMINISTRATION",
        "Admin Control Center",
        "Manage registered accounts, roles, access status and security actions.",
    )

    with get_db_connection() as connection:
        total_users = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        active_users = connection.execute("SELECT COUNT(*) FROM users WHERE status='active'").fetchone()[0]
        disabled_users = connection.execute("SELECT COUNT(*) FROM users WHERE status='disabled'").fetchone()[0]
        total_admins = connection.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]

    st.markdown(
        """
        <div class="admin-hero">
            <div class="admin-kicker">SECURE USER MANAGEMENT</div>
            <div class="admin-title">👑 AcademicAI Administration</div>
            <div class="admin-sub">Review accounts, control access and maintain the local authentication database.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("👥 Total Users", total_users)
    m2.metric("🟢 Active", active_users)
    m3.metric("🚫 Disabled", disabled_users)
    m4.metric("👑 Admins", total_admins)

    st.write("")
    tab_users, tab_create, tab_logs = st.tabs(["👥 User Management", "➕ Create User", "📜 Activity Log"])

    with tab_users:
        search = st.text_input(
            "Search users",
            placeholder="Search by User ID, name or email...",
            key="admin_user_search",
        )
        users = get_all_users(search)

        if not users:
            st.info("No users found.")
        else:
            table_data = pd.DataFrame([dict(row) for row in users])
            table_data.columns = [c.replace("_", " ").title() for c in table_data.columns]
            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Role": st.column_config.TextColumn("Role"),
                    "Status": st.column_config.TextColumn("Status"),
                },
            )

            st.markdown("### Manage an account")
            user_ids = [row["user_id"] for row in users]
            selected_id = st.selectbox("Select User ID", user_ids, key="admin_selected_user")
            selected = next(row for row in users if row["user_id"] == selected_id)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Name", selected["full_name"])
            c2.metric("Role", selected["role"].title())
            c3.metric("Status", selected["status"].title())
            c4.metric("Last Login", selected["last_login_at"] or "Never")

            is_self = selected["user_id"].lower() == st.session_state.get("username", "").lower()
            st.markdown(f"**Email:** {selected['email']}")
            st.caption(f"Created: {selected['created_at']}")

            a, b, c = st.columns(3)
            with a:
                if selected["status"] == "active":
                    if st.button("🚫 Disable Account", use_container_width=True, disabled=is_self, key=f"disable_{selected_id}"):
                        update_user_status(selected_id, "disabled")
                        admin_log("DISABLE_USER", selected_id, "Account disabled")
                        st.success("Account disabled.")
                        st.rerun()
                else:
                    if st.button("🟢 Enable Account", use_container_width=True, key=f"enable_{selected_id}"):
                        update_user_status(selected_id, "active")
                        admin_log("ENABLE_USER", selected_id, "Account enabled")
                        st.success("Account enabled.")
                        st.rerun()

            with b:
                if selected["role"] == "user":
                    if st.button("👑 Make Admin", use_container_width=True, disabled=is_self, key=f"promote_{selected_id}"):
                        update_user_role(selected_id, "admin")
                        admin_log("PROMOTE_USER", selected_id, "Role changed to admin")
                        st.success("User promoted to admin.")
                        st.rerun()
                else:
                    can_demote = not is_self and count_admins() > 1
                    if st.button("👤 Remove Admin", use_container_width=True, disabled=not can_demote, key=f"demote_{selected_id}"):
                        update_user_role(selected_id, "user")
                        admin_log("DEMOTE_USER", selected_id, "Role changed to user")
                        st.success("Admin role removed.")
                        st.rerun()

            with c:
                delete_confirm = st.checkbox(
                    "Confirm delete",
                    key=f"confirm_delete_{selected_id}",
                    disabled=is_self,
                )
                if st.button("🗑️ Delete Account", use_container_width=True, disabled=(is_self or not delete_confirm), key=f"delete_{selected_id}"):
                    delete_user(selected_id)
                    admin_log("DELETE_USER", selected_id, "Account permanently deleted")
                    st.success("Account permanently deleted.")
                    st.rerun()

            with st.expander("🔐 Reset User Password"):
                new_password = st.text_input(
                    "New Password",
                    type="password",
                    placeholder="8+ chars, uppercase, lowercase and number",
                    key=f"reset_pw_{selected_id}",
                )
                confirm_new = st.text_input(
                    "Confirm New Password",
                    type="password",
                    key=f"reset_pw_confirm_{selected_id}",
                )
                if st.button("Reset Password", type="primary", key=f"reset_btn_{selected_id}"):
                    if not password_is_strong(new_password):
                        st.error("Password must contain 8+ characters, uppercase, lowercase and a number.")
                    elif new_password != confirm_new:
                        st.error("Passwords do not match.")
                    else:
                        reset_user_password(selected_id, new_password)
                        admin_log("RESET_PASSWORD", selected_id, "Password reset by administrator")
                        st.success("Password reset successfully.")

    with tab_create:
        st.markdown("### Create a new account")
        st.caption("Admin-created accounts use the same secure password hashing as normal registration.")
        ac1, ac2 = st.columns(2)
        with ac1:
            admin_full_name = st.text_input("Full Name", key="admin_create_name")
            admin_user_id = st.text_input("User ID", key="admin_create_user")
        with ac2:
            admin_email = st.text_input("Email Address", key="admin_create_email")
            admin_password = st.text_input("Temporary Password", type="password", key="admin_create_password")

        make_admin = st.checkbox("Create this account as Admin", key="admin_create_role")
        if st.button("Create Account", type="primary", use_container_width=True):
            uid = normalize_user_id(admin_user_id)
            email = normalize_email(admin_email)
            if not admin_full_name.strip() or not valid_user_id(uid) or not valid_email(email) or not password_is_strong(admin_password):
                st.error("Enter a valid name, User ID, email and strong password.")
            else:
                ok, msg = create_user(uid, email, admin_full_name.strip(), admin_password)
                if ok:
                    if make_admin:
                        update_user_role(uid, "admin")
                    admin_log("CREATE_USER", uid, "Admin created account")
                    st.success("Account created successfully.")
                    st.rerun()
                else:
                    st.error(msg)

    with tab_logs:
        with get_db_connection() as connection:
            logs = connection.execute(
                "SELECT created_at, admin_user_id, action, target_user_id, details FROM admin_activity ORDER BY id DESC LIMIT 200"
            ).fetchall()
        if logs:
            logs_df = pd.DataFrame([dict(row) for row in logs])
            st.dataframe(logs_df, use_container_width=True, hide_index=True)
        else:
            st.info("No admin activity recorded yet.")

# =========================================================
# ABOUT
# =========================================================
elif st.session_state.page == "ℹ️ About":
    page_header(
        "PROJECT INFORMATION",
        "About AcademicAI",
        "A Machine Learning based student performance analytics system.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Student Records", f"{len(df):,}")
    c2.metric("Features", df.shape[1])
    c3.metric("Predictors", 10)
    c4.metric("Model", "Linear Regression")

    st.write("")

    left, right = st.columns([1.4, .8])

    with left:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="color:#111827;">🎓 Project Overview</h3>
                <p class="muted" style="line-height:1.8;">
                    AcademicAI uses Machine Learning and data analytics to
                    analyze student academic performance, predict expected
                    final marks and provide performance insights.
                </p>
                <h4 style="color:#111827;">Core capabilities</h4>
                <p class="muted" style="line-height:1.9;">
                    ✓ Student performance prediction<br>
                    ✓ Dataset exploration and visualization<br>
                    ✓ Feature coefficient analysis<br>
                    ✓ Model evaluation<br>
                    ✓ Academic goal planning<br>
                    ✓ Personalized recommendations
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="color:#111827;">🛠 Technology Stack</h3>
                <p class="muted" style="line-height:2;">
                    🐍 Python<br>
                    🐼 Pandas<br>
                    🔢 NumPy<br>
                    🤖 Scikit-Learn<br>
                    📊 Plotly<br>
                    🌐 Streamlit<br>
                    💾 Joblib
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### 🔄 Machine Learning Pipeline")

    pipeline = [
        ("01", "Data Collection", "Student academic and learning-related data"),
        ("02", "Preprocessing", "StandardScaler + OneHotEncoder"),
        ("03", "Pipeline", "ColumnTransformer + Linear Regression"),
        ("04", "Evaluation", "R², MAE, MSE and RMSE"),
        ("05", "Persistence", "Saved with Joblib"),
        ("06", "Deployment", "Interactive Streamlit application"),
    ]

    for start in range(0, len(pipeline), 3):
        row = st.columns(3)
        for col, (num, title, text) in zip(row, pipeline[start:start + 3]):
            with col:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div style="color:#635bff;font-weight:800;">{num}</div>
                        <h4 style="color:#111827;">{title}</h4>
                        <div class="muted" style="line-height:1.6;">{text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.warning(
        "Predicted marks are estimated outputs from the trained Machine Learning "
        "model and should not be treated as official academic grades."
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        <b>🎓 AcademicAI — Student Performance Prediction & Analytics</b><br>
        Machine Learning • Data Analytics • Academic Insights
    </div>
    """,
    unsafe_allow_html=True,
)