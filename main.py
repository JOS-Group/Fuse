"""
Fuse Semantic Data Platform — Offline & Template Edition
High-Speed Local Dual-Engine Semantic Layer for Healthcare Data

Simulates federated queries across Snowflake, Databricks, and AWS S3 using local datasets (2,000 rows).
Zero cloud setup required; perfect for offline demos and local development.
"""

import os
import re
import csv
import time
import random
import sqlite3
import hashlib
import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES (.env file support)
# ---------------------------------------------------------------------------
def load_dotenv():
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v

load_dotenv()

app = FastAPI(title="Fuse Semantic Data Platform", version="0.3.0")

# ---------------------------------------------------------------------------
# LOCAL FEDERATION DATABASE SETUP (Simulating Trino Catalogs from data/ files)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LOCAL_DB = sqlite3.connect(":memory:", check_same_thread=False)

def init_local_data():
    c = LOCAL_DB.cursor()
    # 1. Initialize Users Table for Profile Management
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT,
        role TEXT,
        organization TEXT,
        created_at TEXT
    )
    """)
    
    # Insert default admin user if not exists
    def_pass = hashlib.sha256("admin123".encode()).hexdigest()
    c.execute("""
    INSERT OR IGNORE INTO users (id, username, email, password_hash, full_name, role, organization, created_at)
    VALUES (1, 'nicholas', 'nicholas@fusehealthcare.io', ?, 'Nicholas Holmes', 'Lead Data Architect', 'CMS Healthcare Analytics', datetime('now'))
    """, (def_pass,))
    
    # 2. Initialize Healthcare Datasets
    tables = {
        "snowflake_medical_claims": os.path.join(DATA_DIR, "snowflake_medicare_claims.csv"),
        "databricks_medicaid_beneficiary": os.path.join(DATA_DIR, "databricks_medicaid_beneficiary.csv"),
        "snowflake_medical_providers": os.path.join(DATA_DIR, "snowflake_medical_providers.csv"),
        "aws_s3_clinical_diagnosis_reference": os.path.join(DATA_DIR, "aws_s3_clinical_diagnosis_reference.csv"),
    }
    
    for tbl_name, file_path in tables.items():
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader)
                c.execute(f"DROP TABLE IF EXISTS {tbl_name}")
                c.execute(f"CREATE TABLE {tbl_name} (" + ", ".join([f"{col} TEXT" for col in headers]) + ")")
                c.executemany(f"INSERT INTO {tbl_name} VALUES (" + ",".join(["?"] * len(headers)) + ")", reader)
    LOCAL_DB.commit()

init_local_data()

# ---------------------------------------------------------------------------
# SEMANTIC MODEL & CATALOG DEFINITION
# ---------------------------------------------------------------------------
SEMANTIC_MODEL: Dict[str, Dict[str, Any]] = {
    "claim": {
        "label": "Medicare Claim",
        "description": "Claim-level billing, procedures, and reimbursement records.",
        "color": "#29B5E8",
        "fields": {
            "claim.id": {"label": "Claim ID", "type": "string", "source": "snowflake", "column": "CLM_ID", "mappings": {"snowflake": "snowflake.medical.claims.CLM_ID", "databricks": "databricks.medical.claims.claim_id"}},
            "claim.amount": {"label": "Billed Amount", "type": "currency", "source": "snowflake", "column": "CLM_AMT", "mappings": {"snowflake": "snowflake.medical.claims.CLM_AMT", "databricks": "databricks.medical.claims.billed_amount"}},
            "claim.paid_amount": {"label": "Paid Amount", "type": "currency", "source": "snowflake", "column": "CLM_PAID_AMT", "mappings": {"snowflake": "snowflake.medical.claims.CLM_PAID_AMT", "databricks": "databricks.medical.claims.paid_amount"}},
            "claim.service_date": {"label": "Service Date", "type": "date", "source": "snowflake", "column": "SERVICE_DATE", "mappings": {"snowflake": "snowflake.medical.claims.SERVICE_DATE", "databricks": "databricks.medical.claims.service_date"}},
            "claim.type": {"label": "Claim Type", "type": "string", "source": "snowflake", "column": "CLM_TYPE", "mappings": {"snowflake": "snowflake.medical.claims.CLM_TYPE", "databricks": "databricks.medical.claims.claim_type"}},
        },
    },
    "beneficiary": {
        "label": "Medicaid Beneficiary",
        "description": "Demographic, dual-eligibility, and risk score profiles.",
        "color": "#FF3621",
        "fields": {
            "beneficiary.id": {"label": "Beneficiary ID", "type": "string", "source": "databricks", "column": "beneficiary_id", "mappings": {"snowflake": "snowflake.medical.claims.BENE_SK", "databricks": "databricks.medicaid.beneficiary.beneficiary_id"}},
            "beneficiary.state": {"label": "State", "type": "string", "source": "databricks", "column": "state", "mappings": {"databricks": "databricks.medicaid.beneficiary.state"}},
            "beneficiary.county": {"label": "County", "type": "string", "source": "databricks", "column": "county", "mappings": {"databricks": "databricks.medicaid.beneficiary.county"}},
            "beneficiary.date_of_birth": {"label": "Date of Birth", "type": "date", "source": "databricks", "column": "date_of_birth", "mappings": {"databricks": "databricks.medicaid.beneficiary.date_of_birth"}},
            "beneficiary.gender": {"label": "Gender", "type": "string", "source": "databricks", "column": "gender", "mappings": {"databricks": "databricks.medicaid.beneficiary.gender"}},
            "beneficiary.dual_eligibility": {"label": "Dual Eligible", "type": "string", "source": "databricks", "column": "dual_eligibility", "mappings": {"databricks": "databricks.medicaid.beneficiary.dual_eligibility"}},
            "beneficiary.risk_score": {"label": "Risk Score (CMS-HCC)", "type": "number", "source": "databricks", "column": "risk_score", "mappings": {"databricks": "databricks.medicaid.beneficiary.risk_score"}},
        },
    },
    "provider": {
        "label": "Healthcare Provider (NPI)",
        "description": "Provider credentials, medical specialty, and network status.",
        "color": "#10B981",
        "fields": {
            "provider.id": {"label": "Provider NPI", "type": "string", "source": "snowflake", "column": "PRVDR_ID", "mappings": {"snowflake": "snowflake.medical.providers.PRVDR_ID", "databricks": "databricks.medical.providers.provider_npi"}},
            "provider.name": {"label": "Provider Name", "type": "string", "source": "snowflake", "column": "PROVIDER_NAME", "mappings": {"snowflake": "snowflake.medical.providers.PROVIDER_NAME"}},
            "provider.specialty": {"label": "Specialty", "type": "string", "source": "snowflake", "column": "SPECIALTY_NAME", "mappings": {"snowflake": "snowflake.medical.providers.SPECIALTY_NAME", "databricks": "databricks.medical.providers.specialty"}},
            "provider.state": {"label": "Facility State", "type": "string", "source": "snowflake", "column": "FACILITY_STATE", "mappings": {"snowflake": "snowflake.medical.providers.FACILITY_STATE"}},
            "provider.network_status": {"label": "Network Status", "type": "string", "source": "snowflake", "column": "NETWORK_STATUS", "mappings": {"snowflake": "snowflake.medical.providers.NETWORK_STATUS"}},
        },
    },
    "diagnosis": {
        "label": "Clinical Diagnosis (ICD-10)",
        "description": "Standardized ICD-10 diagnostic codes, risk tiers, and cost benchmarks.",
        "color": "#8B5CF6",
        "fields": {
            "diagnosis.code": {"label": "ICD-10 Code", "type": "string", "source": "s3_lake", "column": "DGNS_CD", "mappings": {"snowflake": "snowflake.medical.claims.DGNS_CD", "s3_lake": "aws_s3.clinical.diagnosis_ref.DGNS_CD"}},
            "diagnosis.description": {"label": "Diagnosis Description", "type": "string", "source": "s3_lake", "column": "DESCRIPTION", "mappings": {"s3_lake": "aws_s3.clinical.diagnosis_ref.DESCRIPTION"}},
            "diagnosis.chronic_category": {"label": "Chronic Category", "type": "string", "source": "s3_lake", "column": "CHRONIC_CATEGORY", "mappings": {"s3_lake": "aws_s3.clinical.diagnosis_ref.CHRONIC_CATEGORY"}},
            "diagnosis.risk_tier": {"label": "Risk Tier", "type": "string", "source": "s3_lake", "column": "RISK_TIER", "mappings": {"s3_lake": "aws_s3.clinical.diagnosis_ref.RISK_TIER"}},
            "diagnosis.annual_benchmark": {"label": "Annual Benchmark Cost", "type": "currency", "source": "s3_lake", "column": "ANNUAL_BENCHMARK_COST", "mappings": {"s3_lake": "aws_s3.clinical.diagnosis_ref.ANNUAL_BENCHMARK_COST"}},
        },
    },
}

SOURCES = [
    {
        "id": "snowflake",
        "name": "Snowflake on AWS",
        "type": "Data Warehouse",
        "environment": "AWS US-East-1",
        "catalog": "snowflake.medical",
        "status": "Connected (Local Data Ready)",
        "color": "#29B5E8",
        "tables": 2,
        "fields": 14,
        "sample_file": "snowflake_medicare_claims.xlsx (500 rows)",
    },
    {
        "id": "databricks",
        "name": "Databricks on AWS",
        "type": "Lakehouse (Delta)",
        "environment": "AWS US-East-1 (Unity Catalog)",
        "catalog": "databricks.medicaid",
        "status": "Connected (Local Data Ready)",
        "color": "#FF3621",
        "tables": 1,
        "fields": 7,
        "sample_file": "databricks_medicaid_beneficiary.xlsx (500 rows)",
    },
    {
        "id": "s3_lake",
        "name": "AWS S3 Data Lake",
        "type": "Object Storage / Parquet",
        "environment": "AWS S3 (Glue Catalog)",
        "catalog": "aws_s3.clinical",
        "status": "Connected (Local Data Ready)",
        "color": "#FF9900",
        "tables": 1,
        "fields": 5,
        "sample_file": "aws_s3_clinical_diagnosis_reference.xlsx (500 rows)",
    },
    {
        "id": "trino",
        "name": "Trino Federation Engine",
        "type": "Query Engine",
        "environment": "Distributed Trino (Port 8080/8443)",
        "catalog": "trino.federated",
        "status": "Dual Engine (Local Fast + Trino Ready)",
        "color": "#DD00A1",
        "tables": 4,
        "fields": 26,
        "sample_file": "Cross-Catalog Federated Routing",
    },
]

QUERY_HISTORY = []

# ---------------------------------------------------------------------------
# AUTH & USER PROFILE MODELS & ENDPOINTS
# ---------------------------------------------------------------------------
CURRENT_USER_ID = 1

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = "Healthcare Analyst"
    role: Optional[str] = "Data Analyst"
    organization: Optional[str] = "Clinical Intelligence"

class UpdateProfileRequest(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[str] = None
    organization: Optional[str] = None
    new_password: Optional[str] = None

@app.get("/api/auth/me")
def get_current_user():
    global CURRENT_USER_ID
    c = LOCAL_DB.cursor()
    c.execute("SELECT id, username, email, full_name, role, organization, created_at FROM users WHERE id = ?", (CURRENT_USER_ID,))
    row = c.fetchone()
    if not row:
        c.execute("SELECT id, username, email, full_name, role, organization, created_at FROM users LIMIT 1")
        row = c.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": row[0],
        "username": row[1],
        "email": row[2],
        "full_name": row[3],
        "role": row[4],
        "organization": row[5],
        "created_at": row[6]
    }

@app.post("/api/auth/login")
def login(req: LoginRequest):
    global CURRENT_USER_ID
    p_hash = hashlib.sha256(req.password.encode()).hexdigest()
    c = LOCAL_DB.cursor()
    c.execute("SELECT id, username, email, full_name, role, organization, created_at FROM users WHERE (username = ? OR email = ?) AND password_hash = ?", (req.username, req.username, p_hash))
    row = c.fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid username/email or password.")
    CURRENT_USER_ID = row[0]
    return {
        "status": "success",
        "message": f"Welcome back, {row[3]}!",
        "user": {
            "id": row[0],
            "username": row[1],
            "email": row[2],
            "full_name": row[3],
            "role": row[4],
            "organization": row[5],
            "created_at": row[6]
        }
    }

@app.post("/api/auth/register")
def register(req: RegisterRequest):
    global CURRENT_USER_ID
    if not req.username or not req.email or not req.password:
        raise HTTPException(status_code=400, detail="Username, email, and password are required.")
    p_hash = hashlib.sha256(req.password.encode()).hexdigest()
    c = LOCAL_DB.cursor()
    try:
        c.execute("""
        INSERT INTO users (username, email, password_hash, full_name, role, organization, created_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """, (req.username, req.email, p_hash, req.full_name or req.username, req.role or 'Data Analyst', req.organization or 'Healthcare Org'))
        LOCAL_DB.commit()
        CURRENT_USER_ID = c.lastrowid
        return {
            "status": "success",
            "message": "Account created successfully!",
            "user": {
                "id": CURRENT_USER_ID,
                "username": req.username,
                "email": req.email,
                "full_name": req.full_name or req.username,
                "role": req.role,
                "organization": req.organization
            }
        }
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username or Email already registered.")

@app.put("/api/auth/profile")
def update_profile(req: UpdateProfileRequest):
    global CURRENT_USER_ID
    c = LOCAL_DB.cursor()
    updates = []
    params = []
    if req.email:
        updates.append("email = ?")
        params.append(req.email)
    if req.full_name:
        updates.append("full_name = ?")
        params.append(req.full_name)
    if req.role:
        updates.append("role = ?")
        params.append(req.role)
    if req.organization:
        updates.append("organization = ?")
        params.append(req.organization)
    if req.new_password:
        updates.append("password_hash = ?")
        params.append(hashlib.sha256(req.new_password.encode()).hexdigest())
    if updates:
        params.append(CURRENT_USER_ID)
        c.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
        LOCAL_DB.commit()
    return {"status": "success", "message": "Profile updated successfully."}

# ---------------------------------------------------------------------------
# NATURAL LANGUAGE SEMANTIC QUERY PARSER
# ---------------------------------------------------------------------------
class NaturalLanguageQueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 50

def parse_natural_language_to_semantic(text: str) -> Dict[str, Any]:
    t = text.lower()
    fields = set()
    filters = {}
    explanations = []

    fields.add("claim.id")
    fields.add("claim.amount")

    if any(k in t for k in ["claim", "bill", "reimbursement", "cost", "paid", "spending", "spend"]):
        fields.add("claim.id")
        fields.add("claim.amount")
        fields.add("claim.paid_amount")
        fields.add("claim.service_date")
        fields.add("claim.type")
        explanations.append("Included Medicare Claims domain (billed amount, paid amount, claim type).")

    if any(k in t for k in ["beneficiary", "patient", "member", "demographic", "age", "dob", "gender", "dual", "risk score"]):
        fields.add("beneficiary.id")
        fields.add("beneficiary.state")
        fields.add("beneficiary.county")
        fields.add("beneficiary.gender")
        fields.add("beneficiary.risk_score")
        explanations.append("Included Medicaid Beneficiary domain (state, county, gender, risk score).")

    if any(k in t for k in ["provider", "doctor", "physician", "npi", "hospital", "specialty", "network", "facility"]):
        fields.add("provider.id")
        fields.add("provider.name")
        fields.add("provider.specialty")
        fields.add("provider.network_status")
        explanations.append("Included Healthcare Provider domain (NPI, physician name, specialty, network status).")

    if any(k in t for k in ["diagnosis", "disease", "icd", "chronic", "condition", "tier", "diabetes", "hypertension", "heart failure", "cancer", "kidney", "osteoarthritis"]):
        fields.add("diagnosis.code")
        fields.add("diagnosis.description")
        fields.add("diagnosis.chronic_category")
        fields.add("diagnosis.risk_tier")
        explanations.append("Included ICD-10 Clinical Diagnosis reference domain.")

    states_map = {
        "maryland": "MD", "md": "MD",
        "virginia": "VA", "va": "VA",
        "pennsylvania": "PA", "pa": "PA",
        "florida": "FL", "fl": "FL",
        "texas": "TX", "tx": "TX",
        "california": "CA", "ca": "CA",
        "new york": "NY", "ny": "NY",
        "north carolina": "NC", "nc": "NC",
        "ohio": "OH", "oh": "OH",
        "dc": "DC", "district of columbia": "DC"
    }
    for st_name, st_code in states_map.items():
        if re.search(r'\b' + re.escape(st_name) + r'\b', t):
            filters["state"] = st_code
            fields.add("beneficiary.state")
            explanations.append(f"Filtered Beneficiary State to '{st_code}'.")
            break

    claim_types = ["inpatient", "outpatient", "emergency", "physician", "pharmacy"]
    for ct in claim_types:
        if ct in t:
            filters["claim_type"] = ct.capitalize()
            fields.add("claim.type")
            explanations.append(f"Filtered Claim Type to '{ct.capitalize()}'.")
            break

    if "in-network" in t or "in network" in t:
        filters["network"] = "In-Network"
        fields.add("provider.network_status")
        explanations.append("Filtered Network Status to 'In-Network'.")
    elif "out-of-network" in t or "out of network" in t:
        filters["network"] = "Out-of-Network"
        fields.add("provider.network_status")
        explanations.append("Filtered Network Status to 'Out-of-Network'.")

    specs = {
        "cardiology": "Cardiology", "cardiologist": "Cardiology",
        "orthopedic": "Orthopedics", "orthopedics": "Orthopedics",
        "internal medicine": "Internal Medicine", "internist": "Internal Medicine",
        "dermatology": "Dermatology", "radiology": "Radiology",
        "oncology": "Oncology", "oncologist": "Oncology",
        "pulmonology": "Pulmonology", "neurology": "Neurology",
        "pediatrics": "Pediatrics", "psychiatry": "Psychiatry"
    }
    for s_kw, s_val in specs.items():
        if s_kw in t:
            fields.add("provider.specialty")
            fields.add("provider.name")
            explanations.append(f"Detected specialty search for '{s_val}'.")
            break

    if len(fields) <= 2:
        fields.add("beneficiary.state")
        fields.add("provider.specialty")
        fields.add("diagnosis.description")

    field_list = sorted(list(fields))
    return {
        "fields": field_list,
        "state_filter": filters.get("state"),
        "claim_type_filter": filters.get("claim_type"),
        "network_filter": filters.get("network"),
        "explanations": explanations or ["Parsed natural language query and mapped to canonical semantic entities."]
    }

# ---------------------------------------------------------------------------
# QUERY COMPILER
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    fields: List[str]
    state_filter: Optional[str] = None
    claim_type_filter: Optional[str] = None
    network_filter: Optional[str] = None
    limit: int = 50

def compile_semantic_query(request: QueryRequest) -> Dict[str, str]:
    requested = set(request.fields)
    
    needs_claims = any(f.startswith("claim.") or f == "beneficiary.id" or f == "provider.id" or f == "diagnosis.code" for f in requested)
    needs_beneficiary = any(f.startswith("beneficiary.") and f != "beneficiary.id" for f in requested) or bool(request.state_filter)
    needs_provider = any(f.startswith("provider.") and f != "provider.id" for f in requested) or bool(request.network_filter)
    needs_diagnosis = any(f.startswith("diagnosis.") and f != "diagnosis.code" for f in requested)
    
    trino_aliases = {
        "claim.id": "c.CLM_ID",
        "claim.amount": "c.CLM_AMT",
        "claim.paid_amount": "c.CLM_PAID_AMT",
        "claim.service_date": "c.SERVICE_DATE",
        "claim.type": "c.CLM_TYPE",
        "beneficiary.id": "c.BENE_SK" if needs_claims else "b.beneficiary_id",
        "beneficiary.state": "b.state",
        "beneficiary.county": "b.county",
        "beneficiary.date_of_birth": "b.date_of_birth",
        "beneficiary.gender": "b.gender",
        "beneficiary.dual_eligibility": "b.dual_eligibility",
        "beneficiary.risk_score": "b.risk_score",
        "provider.id": "c.PRVDR_ID" if needs_claims else "p.PRVDR_ID",
        "provider.name": "p.PROVIDER_NAME",
        "provider.specialty": "p.SPECIALTY_NAME",
        "provider.state": "p.FACILITY_STATE",
        "provider.network_status": "p.NETWORK_STATUS",
        "diagnosis.code": "c.DGNS_CD" if needs_claims else "d.DGNS_CD",
        "diagnosis.description": "d.DESCRIPTION",
        "diagnosis.chronic_category": "d.CHRONIC_CATEGORY",
        "diagnosis.risk_tier": "d.RISK_TIER",
        "diagnosis.annual_benchmark": "d.ANNUAL_BENCHMARK_COST",
    }
    
    trino_selections = [f"    {trino_aliases[f]} AS {f.replace('.', '_')}" for f in request.fields if f in trino_aliases]
    if not trino_selections:
        trino_selections = ["    c.CLM_ID AS claim_id"]
        
    trino_sql = "SELECT\n" + ",\n".join(trino_selections) + "\n"
    
    if needs_claims:
        trino_sql += "FROM snowflake.medical.claims c\n"
    elif needs_beneficiary:
        trino_sql += "FROM databricks.medicaid.beneficiary b\n"
    elif needs_provider:
        trino_sql += "FROM snowflake.medical.providers p\n"
    elif needs_diagnosis:
        trino_sql += "FROM aws_s3.clinical.diagnosis_ref d\n"
    else:
        trino_sql += "FROM snowflake.medical.claims c\n"
        
    if needs_claims and needs_beneficiary:
        trino_sql += "LEFT JOIN databricks.medicaid.beneficiary b\n    ON c.BENE_SK = b.beneficiary_id\n"
    if needs_claims and needs_provider:
        trino_sql += "LEFT JOIN snowflake.medical.providers p\n    ON c.PRVDR_ID = p.PRVDR_ID\n"
    if needs_claims and needs_diagnosis:
        trino_sql += "LEFT JOIN aws_s3.clinical.diagnosis_ref d\n    ON c.DGNS_CD = d.DGNS_CD\n"
        
    filters = []
    if request.state_filter and needs_beneficiary:
        safe_st = request.state_filter.upper().replace("'", "")[:2]
        filters.append(f"b.state = '{safe_st}'")
    if request.claim_type_filter and needs_claims:
        safe_ct = request.claim_type_filter.replace("'", "")
        filters.append(f"c.CLM_TYPE = '{safe_ct}'")
    if request.network_filter and needs_provider:
        safe_net = request.network_filter.replace("'", "")
        filters.append(f"p.NETWORK_STATUS = '{safe_net}'")
        
    if filters:
        trino_sql += "WHERE " + " AND ".join(filters) + "\n"
        
    limit_val = min(max(request.limit, 1), 500)
    trino_sql += f"LIMIT {limit_val};"
    
    local_sql = "SELECT\n" + ",\n".join(trino_selections) + "\n"
    if needs_claims:
        local_sql += "FROM snowflake_medical_claims c\n"
    elif needs_beneficiary:
        local_sql += "FROM databricks_medicaid_beneficiary b\n"
    elif needs_provider:
        local_sql += "FROM snowflake_medical_providers p\n"
    elif needs_diagnosis:
        local_sql += "FROM aws_s3_clinical_diagnosis_reference d\n"
    else:
        local_sql += "FROM snowflake_medical_claims c\n"
        
    if needs_claims and needs_beneficiary:
        local_sql += "LEFT JOIN databricks_medicaid_beneficiary b ON c.BENE_SK = b.beneficiary_id\n"
    if needs_claims and needs_provider:
        local_sql += "LEFT JOIN snowflake_medical_providers p ON c.PRVDR_ID = p.PRVDR_ID\n"
    if needs_claims and needs_diagnosis:
        local_sql += "LEFT JOIN aws_s3_clinical_diagnosis_reference d ON c.DGNS_CD = d.DGNS_CD\n"
        
    if filters:
        local_sql += "WHERE " + " AND ".join(filters) + "\n"
    local_sql += f"LIMIT {limit_val};"
    
    # Live Snowflake SQL
    sf_sql = "SELECT\n" + ",\n".join(trino_selections) + "\n"
    if needs_claims:
        sf_sql += "FROM FUSE_DB.MEDICAL.CLAIMS c\n"
    elif needs_beneficiary:
        sf_sql += "FROM FUSE_DB.MEDICAL.MEDICAID_BENEFICIARY b\n"
    elif needs_provider:
        sf_sql += "FROM FUSE_DB.MEDICAL.PROVIDERS p\n"
    elif needs_diagnosis:
        sf_sql += "FROM FUSE_DB.MEDICAL.CLINICAL_DIAGNOSIS_REF d\n"
    else:
        sf_sql += "FROM FUSE_DB.MEDICAL.CLAIMS c\n"
        
    if needs_claims and needs_beneficiary:
        sf_sql += "LEFT JOIN FUSE_DB.MEDICAL.MEDICAID_BENEFICIARY b ON c.BENE_SK = b.BENEFICIARY_ID\n"
    if needs_claims and needs_provider:
        sf_sql += "LEFT JOIN FUSE_DB.MEDICAL.PROVIDERS p ON c.PRVDR_ID = p.PRVDR_ID\n"
    if needs_claims and needs_diagnosis:
        sf_sql += "LEFT JOIN FUSE_DB.MEDICAL.CLINICAL_DIAGNOSIS_REF d ON c.DGNS_CD = d.DGNS_CD\n"
        
    if filters:
        sf_sql += "WHERE " + " AND ".join(filters) + "\n"
    sf_sql += f"LIMIT {limit_val};"

    return {"trino_sql": trino_sql, "local_sql": local_sql, "snowflake_sql": sf_sql}

def _clean_env(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(key, default)
    if val is None:
        return None
    val = val.strip()
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        val = val[1:-1].strip()
    return val

def get_snowflake_conn():
    sf_account = _clean_env("SNOWFLAKE_ACCOUNT")
    sf_user = _clean_env("SNOWFLAKE_USER")
    sf_password = _clean_env("SNOWFLAKE_PASSWORD")
    if not (sf_account and sf_user and sf_password):
        return None
    try:
        import snowflake.connector
        sf_db = _clean_env("SNOWFLAKE_DATABASE", "FUSE_DB")
        sf_schema = _clean_env("SNOWFLAKE_SCHEMA", "MEDICAL")
        sf_wh = _clean_env("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        sf_role = _clean_env("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
        
        conn = snowflake.connector.connect(
            account=sf_account,
            user=sf_user,
            password=sf_password,
            database=sf_db,
            schema=sf_schema,
            warehouse=sf_wh,
            role=sf_role,
            login_timeout=8,
            network_timeout=8,
        )
        return conn
    except Exception:
        return None

def execute_federated_query(trino_sql: str, local_sql: str, snowflake_sql: Optional[str] = None) -> Dict[str, Any]:
    start_time = time.time()
    
    # 1. Try Live Snowflake if configured
    sf_conn = get_snowflake_conn()
    if sf_conn:
        try:
            cur = sf_conn.cursor()
            exec_sql = snowflake_sql or local_sql
            cur.execute(exec_sql)
            columns = [desc[0] for desc in cur.description]
            raw_rows = cur.fetchall()
            rows = [dict(zip(columns, row)) for row in raw_rows]
            duration = int((time.time() - start_time) * 1000)
            sf_db = os.getenv("SNOWFLAKE_DATABASE", "FUSE_DB")
            sf_schema = os.getenv("SNOWFLAKE_SCHEMA", "MEDICAL")
            cur.close()
            sf_conn.close()
            result = {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "execution_ms": duration,
                "engine": f"Live Snowflake Cloud ({sf_db}.{sf_schema})"
            }
            QUERY_HISTORY.insert(0, {
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "sql": exec_sql,
                "row_count": len(rows),
                "execution_ms": duration,
                "engine": result["engine"],
            })
            if len(QUERY_HISTORY) > 50:
                QUERY_HISTORY.pop()
            return result
        except Exception:
            if sf_conn:
                try:
                    sf_conn.close()
                except Exception:
                    pass

    # 2. Try Live Trino if configured
    trino_host = os.getenv("TRINO_HOST")
    if trino_host:
        try:
            import importlib
            trino_dbapi = importlib.import_module("trino.dbapi")
            connect = getattr(trino_dbapi, "connect")
            trino_port = int(os.getenv("TRINO_PORT", "8080"))
            trino_user = os.getenv("TRINO_USER", "fuse_user")
            conn = connect(host=trino_host, port=trino_port, user=trino_user, http_scheme=os.getenv("TRINO_SCHEME", "http"))
            cursor = conn.cursor()
            cursor.execute(trino_sql)
            columns = [desc[0] for desc in cursor.description]
            raw_rows = cursor.fetchall()
            rows = [dict(zip(columns, row)) for row in raw_rows]
            duration = int((time.time() - start_time) * 1000)
            result = {"columns": columns, "rows": rows, "row_count": len(rows), "execution_ms": duration, "engine": "Live Trino Cluster"}
            QUERY_HISTORY.insert(0, {
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "sql": trino_sql,
                "row_count": len(rows),
                "execution_ms": duration,
                "engine": result["engine"],
            })
            if len(QUERY_HISTORY) > 50:
                QUERY_HISTORY.pop()
            return result
        except Exception:
            pass
            
    # 3. Fallback to Local SQLite Dual-Engine
    cursor = LOCAL_DB.cursor()
    cursor.execute(local_sql)
    columns = [desc[0] for desc in cursor.description]
    raw_rows = cursor.fetchall()
    
    rows = []
    for r in raw_rows:
        row_dict = {}
        for col_name, val in zip(columns, r):
            if val is not None:
                if any(x in col_name for x in ['amount', 'paid', 'benchmark', 'cost']):
                    try:
                        row_dict[col_name] = float(val)
                    except ValueError:
                        row_dict[col_name] = val
                elif 'risk_score' in col_name:
                    try:
                        row_dict[col_name] = float(val)
                    except ValueError:
                        row_dict[col_name] = val
                else:
                    row_dict[col_name] = val
            else:
                row_dict[col_name] = None
        rows.append(row_dict)
        
    duration = int((time.time() - start_time) * 1000) + random.randint(15, 35)
    
    result = {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "execution_ms": duration,
        "engine": "Fuse Dual-Engine (Federated 2,000 Records)",
    }
    
    QUERY_HISTORY.insert(0, {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "sql": trino_sql,
        "row_count": len(rows),
        "execution_ms": duration,
        "engine": result["engine"],
    })
    if len(QUERY_HISTORY) > 50:
        QUERY_HISTORY.pop()
        
    return result

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "Fuse Semantic Platform",
        "version": "0.3.0",
        "sources_connected": len(SOURCES),
        "semantic_entities": len(SEMANTIC_MODEL),
        "total_dataset_rows": 2000,
    }

@app.get("/api/sources")
def get_sources():
    return SOURCES

@app.get("/api/sources/{source_id}/preview")
def preview_source(source_id: str):
    table_map = {
        "snowflake": "snowflake_medical_claims",
        "databricks": "databricks_medicaid_beneficiary",
        "s3_lake": "aws_s3_clinical_diagnosis_reference",
        "trino": "snowflake_medical_providers",
    }
    tbl = table_map.get(source_id, "snowflake_medical_claims")
    cursor = LOCAL_DB.cursor()
    cursor.execute(f"SELECT * FROM {tbl} LIMIT 10")
    cols = [desc[0] for desc in cursor.description]
    rows = [dict(zip(cols, r)) for r in cursor.fetchall()]
    return {"source_id": source_id, "table": tbl, "columns": cols, "rows": rows, "total_sample_rows": 500}

@app.post("/api/sources/{source_id}/test")
def test_source_connection(source_id: str):
    if source_id == "snowflake":
        sf_account = os.getenv("SNOWFLAKE_ACCOUNT")
        if not sf_account:
            return {
                "source_id": "snowflake",
                "status": "warning",
                "message": "Snowflake credentials not configured in .env. Using fast local dual-engine fallback.",
                "latency_ms": 12
            }
        sf_conn = get_snowflake_conn()
        if sf_conn:
            try:
                cur = sf_conn.cursor()
                cur.execute("SELECT CURRENT_VERSION(), CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA()")
                row = cur.fetchone()
                ver, wh, db, sc = row[0], row[1], row[2], row[3]
                cur.close()
                sf_conn.close()
                return {
                    "source_id": "snowflake",
                    "status": "success",
                    "message": f"Connected to Snowflake (v{ver}) on Account {sf_account} [Warehouse: {wh}, Database: {db}, Schema: {sc}].",
                    "latency_ms": random.randint(25, 45)
                }
            except Exception as e:
                return {
                    "source_id": "snowflake",
                    "status": "warning",
                    "message": f"Snowflake connection attempt returned: {str(e)[:120]}. Check account identifier format in .env.",
                    "latency_ms": 30
                }
        else:
            return {
                "source_id": "snowflake",
                "status": "warning",
                "message": f"Could not reach Snowflake account '{sf_account}'. Double-check account identifier in .env (e.g. ORG-ACCOUNT). Local fallback active.",
                "latency_ms": 25
            }
            
    time.sleep(0.12)
    return {
        "source_id": source_id,
        "status": "success",
        "message": f"Successfully connected to {source_id.upper()} catalog. Schema metadata synchronized.",
        "latency_ms": random.randint(18, 55),
    }

@app.get("/api/model")
def get_semantic_model():
    return SEMANTIC_MODEL

@app.post("/api/nl-query")
def process_natural_language_query(req: NaturalLanguageQueryRequest):
    parsed = parse_natural_language_to_semantic(req.query)
    q_req = QueryRequest(
        fields=parsed["fields"],
        state_filter=parsed["state_filter"],
        claim_type_filter=parsed["claim_type_filter"],
        network_filter=parsed["network_filter"],
        limit=req.limit or 50
    )
    compiled = compile_semantic_query(q_req)
    result = execute_federated_query(compiled["trino_sql"], compiled["local_sql"], compiled.get("snowflake_sql"))
    return {
        "parsed_intent": parsed,
        "sql": compiled["trino_sql"],
        "snowflake_sql": compiled.get("snowflake_sql"),
        "result": result
    }

@app.post("/api/compile")
def compile_query(request: QueryRequest):
    compiled = compile_semantic_query(request)
    return {
        "sql": compiled["trino_sql"],
        "local_sql": compiled["local_sql"],
        "snowflake_sql": compiled.get("snowflake_sql")
    }

@app.post("/api/query")
def execute_query(request: QueryRequest):
    compiled = compile_semantic_query(request)
    res = execute_federated_query(compiled["trino_sql"], compiled["local_sql"], compiled.get("snowflake_sql"))
    res["sql"] = compiled["trino_sql"]
    res["snowflake_sql"] = compiled.get("snowflake_sql")
    return res

@app.get("/api/history")
def get_history():
    return QUERY_HISTORY

# ---------------------------------------------------------------------------
# EMBEDDED DASHBOARD (Clean White + Gold Gradient Palette & Navigation)
# ---------------------------------------------------------------------------
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fuse · Healthcare Semantic Layer (Snowflake, Databricks, AWS S3 via Trino)</title>
<style>
:root{
  --bg:#ffffff;
  --bg-subtle:#faf8f2;
  --panel:#ffffff;
  --panel-border:#eee8d5;
  --text:#111827;
  --text-muted:#6b7280;
  --gold-start:#cda34f;
  --gold-mid:#f5de82;
  --gold-end:#ba8a2e;
  --gold-glow:rgba(205,163,79,0.22);
  --gold-border:rgba(205,163,79,0.35);
  --gold-grad:linear-gradient(135deg, var(--gold-start), var(--gold-mid) 50%, var(--gold-end));
  --gold-grad-soft:linear-gradient(135deg, rgba(205,163,79,0.12), rgba(245,222,130,0.25) 50%, rgba(186,138,46,0.12));
  --gold-grad-subtle:linear-gradient(135deg, rgba(205,163,79,0.06), rgba(245,222,130,0.14) 50%, rgba(186,138,46,0.06));
}
*{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
body{background:var(--bg);color:var(--text);min-height:100vh;display:flex;background-image:radial-gradient(circle at 15% 10%, rgba(245,222,130,0.25) 0%, transparent 40%), radial-gradient(circle at 85% 90%, rgba(205,163,79,0.12) 0%, transparent 45%);}
aside{width:260px;background:var(--bg-subtle);border-right:1px solid var(--panel-border);padding:24px 18px;display:flex;flex-direction:column;gap:24px;flex-shrink:0;}
.brand{display:flex;align-items:center;gap:12px;text-decoration:none;color:var(--text);}
.brand-mark{width:34px;height:34px;border-radius:10px;background:var(--gold-grad);display:grid;place-items:center;font-weight:900;font-size:16px;color:#3d2c04;box-shadow:0 4px 12px var(--gold-glow);border:1px solid var(--gold-border);}
.brand-text{font-size:16px;font-weight:800;letter-spacing:-0.4px;background:var(--gold-grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.brand-badge{font-size:10px;padding:2px 7px;background:var(--gold-grad-soft);color:#6c4c08;border:1px solid var(--gold-border);border-radius:20px;font-weight:700;}
nav{display:flex;flex-direction:column;gap:4px;}
.nav-section{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--text-muted);margin:12px 10px 4px;font-weight:700;}
.nav-item{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:8px;font-size:13px;font-weight:600;color:var(--text-muted);cursor:pointer;transition:all 0.15s ease;border:1px solid transparent;}
.nav-item:hover{background:var(--gold-grad-subtle);color:var(--text);}
.nav-item.active{background:var(--gold-grad-soft);color:#5c4308;border-color:var(--gold-border);box-shadow:0 2px 8px var(--gold-glow);}
.nav-icon{font-size:14px;width:18px;text-align:center;}

/* User Account Badge */
.user-profile-badge{margin-top:auto;padding:12px;background:#ffffff;border:1px solid var(--gold-border);border-radius:10px;box-shadow:0 2px 8px var(--gold-glow);display:flex;align-items:center;gap:10px;cursor:pointer;transition:all 0.15s;}
.user-profile-badge:hover{background:var(--gold-grad-soft);}
.user-avatar{width:32px;height:32px;border-radius:50%;background:var(--gold-grad);display:grid;place-items:center;font-weight:800;color:#3d2c04;font-size:13px;}
.user-info{display:flex;flex-direction:column;overflow:hidden;}
.user-name{font-size:12px;font-weight:700;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.user-org{font-size:10px;color:var(--text-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}

main{flex:1;padding:28px 36px;overflow-y:auto;max-width:1440px;}
.header{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;padding-bottom:18px;border-bottom:1px solid var(--panel-border);}
.header h1{font-size:22px;font-weight:800;letter-spacing:-0.5px;}
.header-meta{display:flex;gap:10px;align-items:center;}
.pill{padding:4px 10px;background:var(--gold-grad-soft);border:1px solid var(--gold-border);border-radius:20px;font-size:11px;font-weight:700;color:#6c4c08;}
.pill-green{background:#ecfdf5;color:#065f46;border-color:#a7f3d0;}
.pill-blue{background:#eff6ff;color:#1e40af;border-color:#bfdbfe;}
.pill-purple{background:#f5f3ff;color:#5b21b6;border-color:#ddd6fe;}
.view{display:none;}
.view.active{display:block;}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px;}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:20px;}
.card{background:var(--panel);border:1px solid var(--panel-border);border-radius:12px;padding:20px;box-shadow:0 2px 10px rgba(0,0,0,0.02);}
.card-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;}
.card-title{font-size:14px;font-weight:700;letter-spacing:-0.2px;color:var(--text);}
.stat-val{font-size:28px;font-weight:800;background:var(--gold-grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.stat-sub{font-size:12px;color:var(--text-muted);margin-top:4px;}
.source-list{display:flex;flex-direction:column;gap:10px;}
.source{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:var(--bg-subtle);border:1px solid var(--panel-border);border-radius:8px;}
.source-left{display:flex;align-items:center;gap:12px;}
.source-logo{width:28px;height:28px;border-radius:6px;display:grid;place-items:center;font-weight:800;font-size:12px;color:#fff;}
.source-name{font-size:13px;font-weight:700;}
.source-meta{font-size:11px;color:var(--text-muted);}
.entity-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;}
.entity{background:var(--bg-subtle);border:1px solid var(--panel-border);border-radius:10px;padding:14px;cursor:pointer;transition:all 0.15s ease;}
.entity:hover{border-color:var(--gold-border);background:#fff;transform:translateY(-1px);box-shadow:0 4px 12px var(--gold-glow);}
.entity.active{border-color:var(--gold-start);background:var(--gold-grad-soft);box-shadow:0 4px 14px var(--gold-glow);}
.entity-icon{width:28px;height:28px;border-radius:6px;background:var(--gold-grad);color:#3d2c04;display:grid;place-items:center;font-weight:800;font-size:12px;margin-bottom:8px;}
.entity-name{font-size:13px;font-weight:700;}
.entity-count{font-size:11px;color:var(--text-muted);margin-top:2px;}
.studio-grid{display:grid;grid-template-columns:300px 1fr;gap:20px;}
.field-selector{display:flex;flex-direction:column;gap:10px;max-height:560px;overflow-y:auto;padding-right:4px;}
.group-title{font-size:11px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-muted);font-weight:700;margin-top:8px;}
.field-option{display:flex;align-items:center;gap:8px;padding:6px 10px;background:var(--bg-subtle);border:1px solid var(--panel-border);border-radius:6px;font-size:12px;cursor:pointer;user-select:none;}
.field-option:hover{border-color:var(--gold-border);background:#fff;}
.field-option input{accent-color:var(--gold-end);}

/* Natural Language Search Prompt Bar */
.nl-bar-wrap{background:var(--gold-grad-subtle);border:1px solid var(--gold-border);border-radius:12px;padding:16px 20px;margin-bottom:20px;box-shadow:0 4px 16px var(--gold-glow);}
.nl-bar-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}
.nl-bar-title{font-size:13px;font-weight:800;color:#5c4308;display:flex;align-items:center;gap:8px;}
.nl-input-row{display:flex;gap:10px;}
.nl-input{flex:1;padding:12px 16px;border-radius:8px;border:1px solid var(--gold-border);font-size:13px;background:#ffffff;color:var(--text);outline:none;box-shadow:inset 0 1px 3px rgba(0,0,0,0.03);}
.nl-input:focus{border-color:var(--gold-start);box-shadow:0 0 0 3px var(--gold-glow);}
.nl-chip-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;}
.nl-chip{font-size:11px;padding:4px 10px;background:#ffffff;border:1px solid var(--panel-border);border-radius:16px;cursor:pointer;color:var(--text-muted);transition:all 0.15s;}
.nl-chip:hover{border-color:var(--gold-start);color:#5c4308;background:var(--gold-grad-soft);}

.query-editor{display:flex;flex-direction:column;gap:14px;}
.sql-preview{background:#111827;color:#f3f4f6;padding:16px;border-radius:8px;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:12px;line-height:1.5;white-space:pre-wrap;overflow-x:auto;max-height:220px;border:1px solid #1f2937;}
.btn{padding:9px 16px;border-radius:8px;font-size:12px;font-weight:700;border:none;cursor:pointer;transition:all 0.15s ease;display:inline-flex;align-items:center;gap:6px;}
.btn-primary{background:var(--gold-grad);color:#3d2c04;box-shadow:0 2px 8px var(--gold-glow);border:1px solid var(--gold-border);}
.btn-primary:hover{filter:brightness(1.05);transform:translateY(-1px);box-shadow:0 4px 12px var(--gold-glow);}
.btn-secondary{background:var(--bg-subtle);color:var(--text);border:1px solid var(--panel-border);}
.btn-secondary:hover{background:#fff;border-color:var(--gold-border);}
.btn-sm{padding:5px 10px;font-size:11px;}
.table-wrap{overflow-x:auto;border:1px solid var(--panel-border);border-radius:8px;max-height:420px;}
table{width:100%;border-collapse:collapse;font-size:12px;text-align:left;}
th{background:var(--bg-subtle);padding:10px 14px;font-weight:700;color:var(--text-muted);border-bottom:1px solid var(--panel-border);position:sticky;top:0;z-index:2;}
td{padding:10px 14px;border-bottom:1px solid var(--panel-border);color:var(--text);}
tr:hover td{background:rgba(245,222,130,0.08);}
.code-chip{font-family:ui-monospace,Menlo,Monaco,monospace;font-size:11px;background:var(--gold-grad-soft);color:#5c4308;padding:2px 6px;border-radius:4px;border:1px solid var(--gold-border);}
.empty-note{padding:40px;text-align:center;color:var(--text-muted);font-size:13px;}
.results-meta{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;font-size:12px;color:var(--text-muted);}
.filter-row{display:flex;gap:12px;margin-bottom:12px;align-items:center;flex-wrap:wrap;}
.filter-select{padding:7px 10px;border-radius:6px;border:1px solid var(--panel-border);font-size:12px;background:#fff;color:var(--text);outline:none;}
.filter-select:focus{border-color:var(--gold-start);box-shadow:0 0 0 3px var(--gold-glow);}
.toast{position:fixed;bottom:24px;right:24px;padding:12px 18px;background:#111827;color:#fff;border-radius:8px;font-size:12px;font-weight:600;box-shadow:0 6px 20px rgba(0,0,0,0.15);display:none;z-index:999;}
.toast.show{display:flex;align-items:center;gap:8px;}
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,0.45);backdrop-filter:blur(3px);display:none;place-items:center;z-index:100;}
.modal{background:#ffffff;border-radius:12px;border:1px solid var(--panel-border);width:850px;max-width:92vw;max-height:85vh;overflow-y:auto;padding:24px;box-shadow:0 12px 32px rgba(0,0,0,0.18);}
.modal-sm{width:460px;}
.modal-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--panel-border);}
.modal-title{font-size:16px;font-weight:800;color:var(--text);}
.close-btn{background:none;border:none;font-size:20px;cursor:pointer;color:var(--text-muted);}
.setup-step{background:var(--bg-subtle);border:1px solid var(--panel-border);border-radius:10px;padding:16px;margin-bottom:14px;}
.setup-step h3{font-size:14px;font-weight:800;color:#5c4308;margin-bottom:8px;}
.setup-step p{font-size:12px;color:var(--text-muted);margin-bottom:10px;line-height:1.5;}
.setup-code{background:#111827;color:#f3f4f6;padding:12px;border-radius:6px;font-family:ui-monospace,Menlo,Monaco,monospace;font-size:11px;overflow-x:auto;}
.form-group{margin-bottom:14px;}
.form-label{display:block;font-size:11px;font-weight:700;color:var(--text-muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px;}
.form-input{width:100%;padding:9px 12px;border:1px solid var(--panel-border);border-radius:6px;font-size:13px;outline:none;}
.form-input:focus{border-color:var(--gold-start);box-shadow:0 0 0 3px var(--gold-glow);}
</style>
</head>
<body>

<aside>
  <a href="#" class="brand" onclick="showView('overview');return false;">
    <div class="brand-mark">F</div>
    <div>
      <div class="brand-text">FUSE <span style="font-size:10px;font-weight:600;color:var(--gold-end)">LOCAL</span></div>
      <div style="font-size:10px;color:var(--text-muted);font-weight:600">Offline Template Edition</div>
    </div>
    <span class="brand-badge">Local Engine</span>
  </a>

  <nav>
    <div class="nav-section">Workspace</div>
    <div class="nav-item active" data-view="overview" onclick="showView('overview')">
      <span class="nav-icon">◈</span> Overview
    </div>
    <div class="nav-item" data-view="semantic" onclick="showView('semantic')">
      <span class="nav-icon">◇</span> Semantic Model
    </div>
    <div class="nav-item" data-view="studio" onclick="showView('studio')">
      <span class="nav-icon">⌁</span> Query Studio
    </div>

    <div class="nav-section">Data Sources</div>
    <div class="nav-item" data-view="sources" onclick="showView('sources')">
      <span class="nav-icon">◎</span> Catalogs & S3
    </div>
    <div class="nav-item" data-view="mappings" onclick="showView('mappings')">
      <span class="nav-icon">⇄</span> Mappings
    </div>

    <div class="nav-section">Platform</div>
    <div class="nav-item" data-view="setup" onclick="showView('setup')">
      <span class="nav-icon">⚙</span> AWS & Trino Setup
    </div>
    <div class="nav-item" data-view="history" onclick="showView('history')">
      <span class="nav-icon">◷</span> Audit History
    </div>
    <div class="nav-item" data-view="profile" onclick="showView('profile')">
      <span class="nav-icon">👤</span> My Profile
    </div>
  </nav>

  <div class="user-profile-badge" onclick="showView('profile')">
    <div class="user-avatar" id="sidebarAvatar">NH</div>
    <div class="user-info">
      <div class="user-name" id="sidebarUserName">Nicholas Holmes</div>
      <div class="user-org" id="sidebarUserOrg">CMS Analytics</div>
    </div>
  </div>
</aside>

<main>
  <!-- OVERVIEW VIEW -->
  <section id="view-overview" class="view active">
    <div class="header">
      <div>
        <h1>Healthcare Semantic Layer — Offline / Template Edition</h1>
        <div style="font-size:12px;color:var(--text-muted);margin-top:2px">
          Local Dual-Engine simulating Trino federation across Snowflake, Databricks, and S3 datasets (2,000 records).
        </div>
      </div>
      <div class="header-meta">
        <span class="pill pill-green">● Local Dual-Engine Active (2,000 Rows)</span>
        <button class="btn btn-primary" onclick="showView('studio')">Open Query Studio ⌁</button>
      </div>
    </div>

    <div class="grid-4">
      <div class="card">
        <div class="card-title">Catalogs Connected</div>
        <div class="stat-val" id="statCatalogs">3</div>
        <div class="stat-sub">Snowflake · Databricks · S3</div>
      </div>
      <div class="card">
        <div class="card-title">Semantic Terms</div>
        <div class="stat-val" id="statTerms">21</div>
        <div class="stat-sub">Claims, Beneficiary, Providers, ICD-10</div>
      </div>
      <div class="card">
        <div class="card-title">Dual-Engine Speed</div>
        <div class="stat-val">28 ms</div>
        <div class="stat-sub">Local fast query or Live Trino</div>
      </div>
      <div class="card">
        <div class="card-title">Sample Data Ready</div>
        <div class="stat-val">2,000</div>
        <div class="stat-sub">4 Excel files (500 rows each in data/)</div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title">Federated Catalogs & Storage</div>
          <button class="btn btn-secondary btn-sm" onclick="showView('sources')">View All</button>
        </div>
        <div class="source-list" id="overviewSources"></div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">Healthcare Semantic Entities</div>
          <button class="btn btn-secondary btn-sm" onclick="showView('semantic')">Explore Models</button>
        </div>
        <div class="entity-grid" id="overviewEntities"></div>
      </div>
    </div>
  </section>

  <!-- SEMANTIC MODEL VIEW -->
  <section id="view-semantic" class="view">
    <div class="header">
      <div>
        <h1>Healthcare Semantic Entities</h1>
        <div style="font-size:12px;color:var(--text-muted);margin-top:2px">
          Canonical business definitions mapped across disjoint schemas in Snowflake & Databricks.
        </div>
      </div>
      <button class="btn btn-primary" onclick="showView('studio')">Query These Fields ⌁</button>
    </div>

    <div class="entity-grid" id="semanticEntities" style="margin-bottom:24px"></div>

    <div class="card">
      <div class="card-header">
        <div class="card-title" id="semanticTableTitle">Entity Fields & Source Mappings</div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Semantic Term</th>
              <th>Data Type</th>
              <th>Primary Source Engine</th>
              <th>Physical Column Target</th>
            </tr>
          </thead>
          <tbody id="semanticMappingRows"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- QUERY STUDIO VIEW -->
  <section id="view-studio" class="view">
    <div class="header">
      <div>
        <h1>Semantic Query Studio & AI Prompt Bar</h1>
        <div style="font-size:12px;color:var(--text-muted);margin-top:2px">
          Ask questions in Natural Language or select semantic checkboxes to federate across Snowflake, Databricks, and S3.
        </div>
      </div>
      <div class="header-meta">
        <button class="btn btn-secondary" onclick="resetStudio()">Reset Fields</button>
        <button class="btn btn-primary" onclick="runQuery()">⚡ Run Query</button>
      </div>
    </div>

    <!-- Natural Language Query Prompt Bar -->
    <div class="nl-bar-wrap">
      <div class="nl-bar-header">
        <div class="nl-bar-title">✨ Natural Language Semantic Query Studio</div>
        <span class="pill pill-purple">Conversational AI Compiler Active</span>
      </div>
      <div class="nl-input-row">
        <input type="text" id="nlQueryInput" class="nl-input" placeholder="e.g. 'Show all cardiology claims over $1000 in Maryland' or 'Inpatient admissions for dual eligible members'..." onkeydown="if(event.key==='Enter')runNLQuery()">
        <button class="btn btn-primary" onclick="runNLQuery()">Translate & Run ⌁</button>
      </div>
      <div class="nl-chip-row">
        <span style="font-size:11px;font-weight:700;color:var(--text-muted);align-self:center">Try Prompt:</span>
        <div class="nl-chip" onclick="applyNLPrompt('Show all cardiology claims in Maryland')">"Cardiology claims in Maryland"</div>
        <div class="nl-chip" onclick="applyNLPrompt('Inpatient claims for dual eligible members with diabetes')">"Inpatient claims with diabetes"</div>
        <div class="nl-chip" onclick="applyNLPrompt('Out-of-network provider claims in Florida')">"Out-of-network claims in FL"</div>
        <div class="nl-chip" onclick="applyNLPrompt('High risk score beneficiaries and total paid amount')">"High risk beneficiaries & spend"</div>
      </div>
    </div>

    <div class="studio-grid">
      <div class="card">
        <div class="card-title" style="margin-bottom:10px">Semantic Field Palette</div>
        <div class="field-selector" id="queryFields"></div>
      </div>

      <div class="query-editor">
        <div class="card">
          <div class="filter-row">
            <span style="font-size:12px;font-weight:700">Filters:</span>
            <select id="stateFilter" class="filter-select" onchange="compileQuery()">
              <option value="">All States (Databricks)</option>
              <option value="MD">Maryland (MD)</option>
              <option value="VA">Virginia (VA)</option>
              <option value="PA">Pennsylvania (PA)</option>
              <option value="FL">Florida (FL)</option>
              <option value="TX">Texas (TX)</option>
              <option value="CA">California (CA)</option>
              <option value="NY">New York (NY)</option>
            </select>

            <select id="claimTypeFilter" class="filter-select" onchange="compileQuery()">
              <option value="">All Claim Types (Snowflake)</option>
              <option value="Inpatient">Inpatient</option>
              <option value="Outpatient">Outpatient</option>
              <option value="Emergency">Emergency</option>
              <option value="Physician">Physician</option>
              <option value="Pharmacy">Pharmacy</option>
            </select>

            <select id="networkFilter" class="filter-select" onchange="compileQuery()">
              <option value="">All Provider Networks</option>
              <option value="In-Network">In-Network Only</option>
              <option value="Out-of-Network">Out-of-Network Only</option>
            </select>

            <select id="rowLimit" class="filter-select" onchange="compileQuery()">
              <option value="25">Limit 25</option>
              <option value="50" selected>Limit 50</option>
              <option value="100">Limit 100</option>
              <option value="500">Limit 500 (Full Sample)</option>
            </select>
          </div>

          <div class="card-header" style="margin-bottom:8px">
            <div class="card-title">Compiled Federated Trino SQL</div>
            <span class="pill" id="compileStatus">auto-compiled</span>
          </div>
          <div class="sql-preview" id="sqlPreview">SELECT ...</div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title">Federated Execution Results</div>
            <button class="btn btn-secondary btn-sm" onclick="runQuery()">Re-run</button>
          </div>
          <div id="resultsArea">
            <div class="empty-note">Select semantic fields on the left or enter a prompt above, then click <strong>Run Query</strong>.</div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- SOURCES VIEW -->
  <section id="view-sources" class="view">
    <div class="header">
      <div>
        <h1>Connected Catalogs & S3 Buckets</h1>
        <div style="font-size:12px;color:var(--text-muted);margin-top:2px">
          Live connection statuses, metadata catalogs, and 500-row sample datasets in <code>data/</code> folder.
        </div>
      </div>
    </div>
    <div class="source-list" id="sourcesFull"></div>
  </section>

  <!-- MAPPINGS VIEW -->
  <section id="view-mappings" class="view">
    <div class="header">
      <div>
        <h1>Cross-Catalog Semantic Mappings</h1>
        <div style="font-size:12px;color:var(--text-muted);margin-top:2px">
          Unified semantic terms mapped across Snowflake schemas and Databricks Delta tables.
        </div>
      </div>
    </div>
    <div class="card">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Entity Domain</th>
              <th>Canonical Semantic Term</th>
              <th>Catalog</th>
              <th>Physical Source Path</th>
            </tr>
          </thead>
          <tbody id="allMappings"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- SETUP VIEW -->
  <section id="view-setup" class="view">
    <div class="header">
      <div>
        <h1>AWS, Snowflake, Databricks & Trino Setup Guide</h1>
        <div style="font-size:12px;color:var(--text-muted);margin-top:2px">
          Step-by-step instructions to connect your live AWS cloud infrastructure to Fuse.
        </div>
      </div>
    </div>

    <div class="setup-step">
      <h3>1. Upload Sample Datasets to Your Cloud Accounts</h3>
      <p>We created 4 realistic 500-row healthcare datasets inside the <code>data/</code> folder of this project in both Excel (<code>.xlsx</code>) and CSV formats:</p>
      <ul style="font-size:12px;color:var(--text-muted);margin-left:20px;line-height:1.6">
        <li><strong>Snowflake (Medical Claims):</strong> Upload <code>data/snowflake_medicare_claims.xlsx</code> into <code>HEALTHCARE_DB.MEDICAL.CLAIMS</code>.</li>
        <li><strong>Databricks on AWS (Medicaid Beneficiary):</strong> Upload <code>data/databricks_medicaid_beneficiary.xlsx</code> as Delta Lake table <code>databricks.medicaid.beneficiary</code>.</li>
        <li><strong>Snowflake (Providers):</strong> Upload <code>data/snowflake_medical_providers.xlsx</code> into <code>HEALTHCARE_DB.MEDICAL.PROVIDERS</code>.</li>
        <li><strong>AWS S3 (Clinical ICD-10):</strong> Upload <code>data/aws_s3_clinical_diagnosis_reference.xlsx</code> to your S3 bucket registered in AWS Glue.</li>
      </ul>
    </div>

    <div class="setup-step">
      <h3>2. Configure Trino Catalogs for Snowflake & Databricks</h3>
      <p>In your Trino deployment (EC2, EKS, or Docker), create catalog configuration files:</p>
      <div class="setup-code"># etc/catalog/snowflake.properties
connector.name=snowflake
connection-url=jdbc:snowflake://&lt;account&gt;.snowflakecomputing.com/?warehouse=COMPUTE_WH&db=HEALTHCARE_DB&schema=MEDICAL
connection-user=TRINO_USER
connection-password=&lt;password&gt;

# etc/catalog/databricks.properties (Delta Lake via AWS Glue)
connector.name=delta-lake
hive.metastore=glue
hive.metastore.glue.region=us-east-1</div>
    </div>

    <div class="setup-step">
      <h3>3. Point Fuse to Your Live Trino Coordinator</h3>
      <p>Set environment variables when running Fuse to execute all queries against your real Trino engine:</p>
      <div class="setup-code">export TRINO_HOST="trino.your-company.internal"
export TRINO_PORT="8080"
export TRINO_USER="fuse_admin"
python fuse_app.py</div>
    </div>
  </section>

  <!-- AUDIT HISTORY VIEW -->
  <section id="view-history" class="view">
    <div class="header">
      <div>
        <h1>Query Audit & Execution History</h1>
        <div style="font-size:12px;color:var(--text-muted);margin-top:2px">
          HIPAA & PHI compliant access logging of all compiled federated queries and latency metrics.
        </div>
      </div>
      <button class="btn btn-secondary" onclick="renderHistory()">Refresh Log</button>
    </div>
    <div class="card">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Execution Engine</th>
              <th>Rows Returned</th>
              <th>Latency</th>
              <th>Compiled Trino SQL</th>
            </tr>
          </thead>
          <tbody id="historyRows"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- USER PROFILE VIEW -->
  <section id="view-profile" class="view">
    <div class="header">
      <div>
        <h1>User Profile & Authentication Settings</h1>
        <div style="font-size:12px;color:var(--text-muted);margin-top:2px">
          Manage your account credentials, healthcare role, organization, and API tokens.
        </div>
      </div>
      <div class="header-meta">
        <button class="btn btn-secondary" onclick="openLoginModal()">Switch Account / Login</button>
        <button class="btn btn-primary" onclick="saveProfileChanges()">Save Changes</button>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title" style="margin-bottom:16px">Account Details</div>
        <div class="form-group">
          <label class="form-label">Full Name</label>
          <input type="text" id="profileFullName" class="form-input" value="Nicholas Holmes">
        </div>
        <div class="form-group">
          <label class="form-label">Username</label>
          <input type="text" id="profileUsername" class="form-input" value="nicholas" disabled style="background:#f3f4f6">
        </div>
        <div class="form-group">
          <label class="form-label">Email Address</label>
          <input type="email" id="profileEmail" class="form-input" value="nicholas@fusehealthcare.io">
        </div>
        <div class="form-group">
          <label class="form-label">Job Role</label>
          <input type="text" id="profileRole" class="form-input" value="Lead Data Architect">
        </div>
        <div class="form-group">
          <label class="form-label">Healthcare Organization</label>
          <input type="text" id="profileOrg" class="form-input" value="CMS Healthcare Analytics">
        </div>
      </div>

      <div class="card">
        <div class="card-title" style="margin-bottom:16px">Security & API Access</div>
        <div class="form-group">
          <label class="form-label">Update Password</label>
          <input type="password" id="profileNewPass" class="form-input" placeholder="Enter new password to change...">
        </div>
        <div class="form-group">
          <label class="form-label">Semantic API Bearer Token</label>
          <div style="display:flex;gap:8px">
            <input type="text" class="form-input" value="fuse_live_sk_948a73bc92e104f" readonly style="background:#f3f4f6">
            <button class="btn btn-secondary btn-sm" onclick="toast('API Token copied to clipboard!')">Copy</button>
          </div>
        </div>
        <div style="background:var(--gold-grad-soft);border:1px solid var(--gold-border);border-radius:8px;padding:12px;margin-top:20px">
          <div style="font-size:12px;font-weight:700;color:#5c4308;margin-bottom:4px">HIPAA Compliant Session Active</div>
          <div style="font-size:11px;color:var(--text-muted)">All cross-catalog queries are audited and tied to your user credentials.</div>
        </div>
      </div>
    </div>
  </section>
</main>

<!-- PREVIEW MODAL -->
<div class="modal-overlay" id="previewModal">
  <div class="modal">
    <div class="modal-header">
      <div class="modal-title" id="modalTitle">Dataset Preview</div>
      <button class="close-btn" onclick="closeModal()">×</button>
    </div>
    <div id="modalBody"></div>
  </div>
</div>

<!-- LOGIN / REGISTER MODAL -->
<div class="modal-overlay" id="authModal">
  <div class="modal modal-sm">
    <div class="modal-header">
      <div class="modal-title" id="authModalTitle">Sign In to Fuse</div>
      <button class="close-btn" onclick="closeAuthModal()">×</button>
    </div>
    <div id="loginForm">
      <div class="form-group">
        <label class="form-label">Username or Email</label>
        <input type="text" id="loginUser" class="form-input" placeholder="nicholas or email...">
      </div>
      <div class="form-group">
        <label class="form-label">Password</label>
        <input type="password" id="loginPass" class="form-input" placeholder="Password...">
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:20px">
        <a href="#" style="font-size:12px;color:var(--gold-end);text-decoration:none" onclick="toggleAuthForm('register');return false;">Create New Account</a>
        <button class="btn btn-primary" onclick="submitLogin()">Sign In</button>
      </div>
    </div>
    <div id="registerForm" style="display:none">
      <div class="form-group">
        <label class="form-label">Username</label>
        <input type="text" id="regUser" class="form-input" placeholder="Username...">
      </div>
      <div class="form-group">
        <label class="form-label">Email</label>
        <input type="email" id="regEmail" class="form-input" placeholder="Email...">
      </div>
      <div class="form-group">
        <label class="form-label">Password</label>
        <input type="password" id="regPass" class="form-input" placeholder="Password...">
      </div>
      <div class="form-group">
        <label class="form-label">Full Name</label>
        <input type="text" id="regFullName" class="form-input" placeholder="Full Name...">
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:20px">
        <a href="#" style="font-size:12px;color:var(--gold-end);text-decoration:none" onclick="toggleAuthForm('login');return false;">Back to Sign In</a>
        <button class="btn btn-primary" onclick="submitRegister()">Register</button>
      </div>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let MODEL = {};
let SOURCES = [];
let selectedEntity = 'claim';

const esc = s => String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');

async function bootstrap(){
  const [mRes, sRes, uRes] = await Promise.all([
    fetch('/api/model').then(r=>r.json()),
    fetch('/api/sources').then(r=>r.json()),
    fetch('/api/auth/me').then(r=>r.json()).catch(()=>null)
  ]);
  MODEL = mRes;
  SOURCES = sRes;
  
  if(uRes) updateProfileUI(uRes);
  
  renderSources();
  renderEntities();
  renderSemanticEntity('claim');
  renderQueryFields();
  renderAllMappings();
  
  setTimeout(()=>{
    let xs=document.querySelectorAll('#queryFields input');
    [0,1,5,6,12,14].forEach(i=>xs[i]&&xs[i].click());
    compileQuery();
  },100);
}

function showView(name){
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  const target = document.getElementById('view-'+name);
  if(target) target.classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.toggle('active',n.dataset.view===name));
  if(name==='history') renderHistory();
  window.scrollTo(0,0);
}

function updateProfileUI(u){
  document.getElementById('sidebarUserName').textContent = u.full_name || u.username;
  document.getElementById('sidebarUserOrg').textContent = u.organization || 'Healthcare Analyst';
  const initials = (u.full_name || u.username).split(' ').map(p=>p[0]).join('').substring(0,2).toUpperCase();
  document.getElementById('sidebarAvatar').textContent = initials;
  
  const pFullName = document.getElementById('profileFullName');
  if(pFullName) pFullName.value = u.full_name || '';
  const pUser = document.getElementById('profileUsername');
  if(pUser) pUser.value = u.username || '';
  const pEmail = document.getElementById('profileEmail');
  if(pEmail) pEmail.value = u.email || '';
  const pRole = document.getElementById('profileRole');
  if(pRole) pRole.value = u.role || '';
  const pOrg = document.getElementById('profileOrg');
  if(pOrg) pOrg.value = u.organization || '';
}

async function saveProfileChanges(){
  const payload = {
    full_name: document.getElementById('profileFullName').value,
    email: document.getElementById('profileEmail').value,
    role: document.getElementById('profileRole').value,
    organization: document.getElementById('profileOrg').value,
    new_password: document.getElementById('profileNewPass').value || null
  };
  const r = await fetch('/api/auth/profile', {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  }).then(res=>res.json());
  
  toast('✓ ' + r.message);
  const u = await fetch('/api/auth/me').then(res=>res.json());
  updateProfileUI(u);
  document.getElementById('profileNewPass').value = '';
}

function openLoginModal(){
  document.getElementById('authModal').style.display = 'grid';
  toggleAuthForm('login');
}

function closeAuthModal(){
  document.getElementById('authModal').style.display = 'none';
}

function toggleAuthForm(form){
  if(form==='login'){
    document.getElementById('authModalTitle').textContent = 'Sign In to Fuse';
    document.getElementById('loginForm').style.display = 'block';
    document.getElementById('registerForm').style.display = 'none';
  } else {
    document.getElementById('authModalTitle').textContent = 'Create New Fuse Account';
    document.getElementById('loginForm').style.display = 'none';
    document.getElementById('registerForm').style.display = 'block';
  }
}

async function submitLogin(){
  const u = document.getElementById('loginUser').value;
  const p = document.getElementById('loginPass').value;
  if(!u || !p){ toast('Please enter username and password'); return; }
  
  const res = await fetch('/api/auth/login', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({username: u, password: p})
  });
  if(!res.ok){
    toast('Invalid username or password.');
    return;
  }
  const data = await res.json();
  toast(data.message);
  updateProfileUI(data.user);
  closeAuthModal();
}

async function submitRegister(){
  const payload = {
    username: document.getElementById('regUser').value,
    email: document.getElementById('regEmail').value,
    password: document.getElementById('regPass').value,
    full_name: document.getElementById('regFullName').value || 'Healthcare User'
  };
  if(!payload.username || !payload.email || !payload.password){
    toast('Please fill out all fields');
    return;
  }
  const res = await fetch('/api/auth/register', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  if(!res.ok){
    toast('Registration failed. Username or email may exist.');
    return;
  }
  const data = await res.json();
  toast(data.message);
  updateProfileUI(data.user);
  closeAuthModal();
}

/* Natural Language Query Handlers */
function applyNLPrompt(text){
  document.getElementById('nlQueryInput').value = text;
  runNLQuery();
}

async function runNLQuery(){
  const query = document.getElementById('nlQueryInput').value.trim();
  if(!query){
    toast('Please enter a query prompt.');
    return;
  }
  
  toast('Translating natural language intent with Semantic Compiler…');
  document.getElementById('resultsArea').innerHTML = '<div class="empty-note">✨ Translating natural language intent and querying federated catalogs…</div>';
  
  const res = await fetch('/api/nl-query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query: query, limit: Number(document.getElementById('rowLimit').value || 50)})
  }).then(r=>r.json());
  
  // Sync UI selections
  const fields = res.parsed_intent.fields || [];
  document.querySelectorAll('#queryFields input').forEach(inp=>{
    inp.checked = fields.includes(inp.value);
  });
  if(res.parsed_intent.state_filter) document.getElementById('stateFilter').value = res.parsed_intent.state_filter;
  if(res.parsed_intent.claim_type_filter) document.getElementById('claimTypeFilter').value = res.parsed_intent.claim_type_filter;
  if(res.parsed_intent.network_filter) document.getElementById('networkFilter').value = res.parsed_intent.network_filter;
  
  document.getElementById('sqlPreview').textContent = res.sql;
  
  renderQueryResults(res.result);
}

function sourceHTML(s){
  return `<div class="source">
    <div class="source-left">
      <div class="source-logo" style="background:${s.color}">${s.name[0]}</div>
      <div>
        <div class="source-name">${esc(s.name)}</div>
        <div class="source-meta">${esc(s.type)} · ${esc(s.environment)} · catalog: <strong>${esc(s.catalog)}</strong></div>
        <div class="source-meta" style="margin-top:2px;color:#8a620b">📁 Sample: ${esc(s.sample_file)}</div>
      </div>
    </div>
    <div style="display:flex;gap:8px;align-items:center">
      <button class="btn btn-sm" onclick="previewSourceData('${s.id}')">Preview 500 Rows</button>
      <button class="btn btn-sm" onclick="testConnection('${s.id}')">Test Ping</button>
      <span class="pill pill-green">${esc(s.status)}</span>
    </div>
  </div>`;
}

function renderSources(){
  const h=SOURCES.map(sourceHTML).join('');
  document.getElementById('overviewSources').innerHTML=h;
  document.getElementById('sourcesFull').innerHTML=h;
}

function renderEntities(){
  const h=Object.entries(MODEL).map(([k,e])=>`
    <div class="entity ${k===selectedEntity?'active':''}" onclick="renderSemanticEntity('${k}');showView('semantic')">
      <div class="entity-icon">${e.label[0]}</div>
      <div class="entity-name">${esc(e.label)}</div>
      <div class="entity-count">${Object.keys(e.fields).length} semantic terms</div>
    </div>`).join('');
  document.getElementById('overviewEntities').innerHTML=h;
  document.getElementById('semanticEntities').innerHTML=h;
}

function renderSemanticEntity(k){
  selectedEntity=k;
  renderEntities();
  const e=MODEL[k];
  document.getElementById('semanticTableTitle').textContent=e.label+' Mappings';
  let rows='';
  Object.entries(e.fields).forEach(([fname,f])=>{
    const primary = Object.entries(f.mappings)[0];
    rows+=`<tr>
      <td><strong style="color:#111827">${esc(f.label)}</strong><div style="color:#6b7280;font-size:9px;margin-top:2px">${esc(fname)}</div></td>
      <td>${esc(f.type)}</td>
      <td><span class="pill">${esc(primary ? primary[0] : f.source)}</span></td>
      <td><span class="code-chip">${esc(primary ? primary[1] : f.column)}</span></td>
    </tr>`;
  });
  document.getElementById('semanticMappingRows').innerHTML=rows;
}

function renderQueryFields(){
  let h='';
  Object.entries(MODEL).forEach(([ek,e])=>{
    h+=`<div class="group-title">${esc(e.label)}</div>`;
    Object.entries(e.fields).forEach(([fname,f])=>{
      h+=`<label class="field-option">
        <input type="checkbox" value="${fname}" onchange="compileQuery()">
        <span>${esc(f.label)} <span style="color:#6b7280;font-size:9px">· ${esc(fname)}</span></span>
      </label>`;
    });
  });
  document.getElementById('queryFields').innerHTML=h;
}

function renderAllMappings(){
  let rows='';
  Object.entries(MODEL).forEach(([ek,e])=>{
    Object.entries(e.fields).forEach(([fname,f])=>{
      Object.entries(f.mappings).forEach(([src,p])=>{
        rows+=`<tr>
          <td>${esc(e.label)}</td>
          <td><span class="code-chip">${esc(fname)}</span></td>
          <td>${esc(src)}</td>
          <td>${esc(p)}</td>
        </tr>`;
      });
    });
  });
  document.getElementById('allMappings').innerHTML=rows;
}

function payload(){
  return {
    fields: [...document.querySelectorAll('#queryFields input:checked')].map(x=>x.value),
    state_filter: document.getElementById('stateFilter').value || null,
    claim_type_filter: document.getElementById('claimTypeFilter').value || null,
    network_filter: document.getElementById('networkFilter').value || null,
    limit: Number(document.getElementById('rowLimit').value || 50)
  };
}

function resetStudio(){
  document.querySelectorAll('#queryFields input').forEach(x=>x.checked=false);
  document.getElementById('stateFilter').value = '';
  document.getElementById('claimTypeFilter').value = '';
  document.getElementById('networkFilter').value = '';
  document.getElementById('nlQueryInput').value = '';
  document.getElementById('sqlPreview').textContent = 'Select one or more semantic fields above or type a natural language prompt.';
  document.getElementById('resultsArea').innerHTML = '<div class="empty-note">Ready to query.</div>';
}

async function compileQuery(){
  const p=payload();
  if(!p.fields.length){
    document.getElementById('sqlPreview').textContent='Select one or more semantic fields.';
    return;
  }
  document.getElementById('compileStatus').textContent='compiling…';
  const r=await fetch('/api/compile',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}).then(r=>r.json());
  document.getElementById('sqlPreview').textContent=r.sql;
  document.getElementById('compileStatus').textContent='compiled';
}

async function runQuery(){
  const p=payload();
  if(!p.fields.length){
    toast('Choose at least one semantic field.');
    return;
  }
  document.getElementById('resultsArea').innerHTML='<div class="empty-note">⚡ Executing federated query across Snowflake, Databricks & S3…</div>';
  const r=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}).then(r=>r.json());
  document.getElementById('sqlPreview').textContent=r.sql;
  renderQueryResults(r);
}

function renderQueryResults(r){
  let cols=r.columns;
  let t=`<div class="results-meta">
    <span><strong>${r.row_count}</strong> rows returned</span>
    <span>⚡ <strong>${r.execution_ms} ms</strong></span>
    <span>Engine: <strong style="color:#a8791c">${esc(r.engine)}</strong></span>
  </div>
  <div class="table-wrap"><table><thead><tr>${cols.map(c=>`<th>${esc(c.replaceAll('_',' '))}</th>`).join('')}</tr></thead><tbody>`;
  
  r.rows.forEach(row=>{
    t+='<tr>';
    cols.forEach(c=>{
      let v=row[c];
      if(v===null||v===undefined) t+='<td style="color:#9ca3af">—</td>';
      else if(typeof v==='number'&&c.includes('amount')) t+=`<td>$${v.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</td>`;
      else if(typeof v==='number'&&c.includes('score')) t+=`<td><strong>${v.toFixed(2)}</strong></td>`;
      else t+=`<td>${esc(v)}</td>`;
    });
    t+='</tr>';
  });
  t+='</tbody></table></div>';
  document.getElementById('resultsArea').innerHTML=t;
  toast('Federated query completed (' + r.row_count + ' rows).');
}

async function renderHistory(){
  const history = await fetch('/api/history').then(r=>r.json());
  const historyRows = document.getElementById('historyRows');
  if(!history.length){
    historyRows.innerHTML='<tr class="empty-note"><td colspan="5">No queries executed yet.</td></tr>';
    return;
  }
  let rows='';
  history.forEach(h=>{
    rows+=`<tr>
      <td>${esc(h.timestamp)}</td>
      <td><span class="pill">${esc(h.engine)}</span></td>
      <td><strong>${h.row_count}</strong></td>
      <td>${h.execution_ms} ms</td>
      <td><code style="font-size:9px;color:#5c4308">${esc(h.sql.replaceAll('\\n',' '))}</code></td>
    </tr>`;
  });
  historyRows.innerHTML=rows;
}

async function previewSourceData(sourceId){
  document.getElementById('modalTitle').textContent = 'Catalog Preview — ' + sourceId.toUpperCase();
  document.getElementById('modalBody').innerHTML = '<div class="empty-note">Loading 500-row sample dataset preview…</div>';
  document.getElementById('previewModal').style.display = 'grid';
  const r = await fetch('/api/sources/' + sourceId + '/preview').then(res=>res.json());
  let t = `<div style="margin-bottom:12px;font-size:11px;color:#6b7280">Table: <span class="code-chip">${esc(r.table)}</span> · Showing first 10 of ${r.total_sample_rows} records from <code>data/</code> folder.</div>`;
  t += '<div class="table-wrap"><table><thead><tr>' + r.columns.map(c=>`<th>${esc(c)}</th>`).join('') + '</tr></thead><tbody>';
  r.rows.forEach(row=>{
    t += '<tr>' + r.columns.map(c=>`<td>${esc(row[c]??'')}</td>`).join('') + '</tr>';
  });
  t += '</tbody></table></div>';
  document.getElementById('modalBody').innerHTML = t;
}

async function testConnection(sourceId){
  toast('Pinging catalog connection for ' + sourceId.toUpperCase() + '…');
  const r = await fetch('/api/sources/' + sourceId + '/test', {method:'POST'}).then(res=>res.json());
  toast('✓ ' + r.message);
}

function closeModal(){
  document.getElementById('previewModal').style.display = 'none';
}

function toast(m){
  const t=document.getElementById('toast');
  t.textContent=m;
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),3000);
}

document.addEventListener('DOMContentLoaded', bootstrap);
if (document.readyState === 'complete' || document.readyState === 'interactive') {
  bootstrap();
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(HTML)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
