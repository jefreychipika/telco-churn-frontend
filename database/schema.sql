-- schema.sql
-- ---------------------------------------------------------------------------
-- Database schema for the Telco Customer Churn app.
--
-- One table is enough for this project: every time someone runs a
-- prediction through the Streamlit dashboard / FastAPI backend, we store
-- the customer details they entered together with the model's verdict.
-- This gives the dashboard a "prediction history" to show, and gives you
-- a real dataset of past predictions if you ever want to retrain later.
--
-- Run this once, before starting the backend, e.g.:
--   mysql -u root -p < schema.sql
-- ---------------------------------------------------------------------------

CREATE DATABASE IF NOT EXISTS telco_churn_db;
USE telco_churn_db;

CREATE TABLE IF NOT EXISTS churn_predictions (
    id                  INT AUTO_INCREMENT PRIMARY KEY,

    -- Demographic / account info
    gender              VARCHAR(10)     NOT NULL,
    senior_citizen      TINYINT(1)      NOT NULL,
    partner             VARCHAR(3)      NOT NULL,   -- 'Yes' / 'No'
    dependents          VARCHAR(3)      NOT NULL,

    -- Account tenure & billing
    tenure              INT             NOT NULL,
    monthly_charges     DECIMAL(8, 2)   NOT NULL,
    total_charges       DECIMAL(10, 2)  NOT NULL,
    contract            VARCHAR(20)     NOT NULL,   -- Month-to-month / One year / Two year
    paperless_billing   VARCHAR(3)      NOT NULL,
    payment_method      VARCHAR(40)     NOT NULL,

    -- Services subscribed to
    phone_service       VARCHAR(3)      NOT NULL,
    multiple_lines      VARCHAR(20)     NOT NULL,
    internet_service    VARCHAR(20)     NOT NULL,
    online_security     VARCHAR(20)     NOT NULL,
    online_backup       VARCHAR(20)     NOT NULL,
    device_protection   VARCHAR(20)     NOT NULL,
    tech_support        VARCHAR(20)     NOT NULL,
    streaming_tv        VARCHAR(20)     NOT NULL,
    streaming_movies    VARCHAR(20)     NOT NULL,

    -- Model output
    prediction          TINYINT(1)      NOT NULL,   -- 1 = will churn, 0 = will stay
    churn_probability   DECIMAL(5, 4)   NOT NULL,   -- e.g. 0.7421

    created_at          TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

-- Speeds up the "recent predictions" / history query used by the dashboard
CREATE INDEX idx_created_at ON churn_predictions (created_at);
