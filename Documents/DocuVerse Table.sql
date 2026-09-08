USE docuverse;

CREATE TABLE employee_pto_balances (
    pto_balance_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    employee_id VARCHAR(50) NOT NULL,
    employee_name VARCHAR(150) NOT NULL,
    pto_type ENUM(
        'VACATION',
        'SICK',
        'PERSONAL',
        'FLOATING_HOLIDAY',
        'OTHER'
    ) NOT NULL,

    balance_year YEAR NOT NULL,
    opening_balance DECIMAL(8,2) NOT NULL DEFAULT 0.00,
    accrued_hours DECIMAL(8,2) NOT NULL DEFAULT 0.00,
    used_hours DECIMAL(8,2) NOT NULL DEFAULT 0.00,
    pending_hours DECIMAL(8,2) NOT NULL DEFAULT 0.00,
    adjusted_hours DECIMAL(8,2) NOT NULL DEFAULT 0.00,

    available_hours DECIMAL(8,2)
        GENERATED ALWAYS AS (
            opening_balance
            + accrued_hours
            + adjusted_hours
            - used_hours
            - pending_hours
        ) STORED,

    last_updated TIMESTAMP NOT NULL
        DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uk_employee_pto_year (
        employee_id,
        pto_type,
        balance_year
    ),

    INDEX idx_employee_id (employee_id)
);

--------------------------------------------------------------

INSERT INTO employee_pto_balances (
    employee_id,
    employee_name,
    pto_type,
    balance_year,
    opening_balance,
    accrued_hours,
    used_hours,
    pending_hours
)
VALUES
(
    'EMP001',
    'John Smith',
    'VACATION',
    2026,
    40.00,
    48.00,
    24.00,
    8.00
),
(
    'EMP001',
    'John Smith',
    'SICK',
    2026,
    16.00,
    32.00,
    8.00,
    0.00
);

--------------------------------------------------------

SELECT
    employee_id,
    employee_name,
    pto_type,
    opening_balance,
    accrued_hours,
    used_hours,
    pending_hours,
    adjusted_hours,
    available_hours
FROM employee_pto_balances
WHERE employee_id = 'EMP001'
  AND balance_year = 2026;
  
--------------------------------------------------------  
  