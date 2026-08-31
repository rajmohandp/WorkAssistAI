SELECT user_name,
    pto_balance_id,
    employee_id,
    employee_name,
    pto_type,
    balance_year	
FROM docuverse.employee_pto_balances

--    available_hours,
--    last_updated,
--    accrued_hours,
--    used_hours,
--    pending_hours,
--    adjusted_hours

-----------------------------------------------------------





USE docuverse;

-- 1. Confirm the active database.
SELECT DATABASE() AS active_database;

-- 2. Confirm the table exists.
SELECT
    table_schema,
    table_name
FROM information_schema.tables
WHERE table_schema = 'docuverse'
  AND table_name = 'employee_pto_balances';

-- 3. Check whether user_name already exists.
SELECT
    column_name,
    column_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'docuverse'
  AND table_name = 'employee_pto_balances'
  AND column_name = 'user_name';



commit;

WHERE employee_id = 'EMP001'
ORDER BY balance_year, pto_type;

-- 5. Count the rows that would receive user_name = 'user01'.
SELECT
    COUNT(*) AS rows_that_would_be_updated
FROM employee_pto_balances
WHERE employee_id = 'EMP001';

-- 6. Check whether the intended username conflicts with existing data.
-- This query works only if user_name already exists.
-- Uncomment it only when Step 3 reports that the column exists.
--
-- SELECT
--     user_name,
--     employee_id,
--     COUNT(*) AS row_count
-- FROM employee_pto_balances
-- WHERE user_name = 'user01'
-- GROUP BY user_name, employee_id;

-- 7. Check for multiple employee names associated with EMP001.
SELECT
    employee_id,
    employee_name,
    COUNT(*) AS row_count
FROM employee_pto_balances
WHERE employee_id = 'EMP001'
GROUP BY employee_id, employee_name;

commit;

------------------------------------------------------------

INSERT INTO docuverse.employee_pto_balances
(
    employee_id,
    employee_name,
    pto_type,
    balance_year,
    opening_balance,
    accrued_hours,
    used_hours,
    pending_hours,
    adjusted_hours,
    last_updated,
    user_name
)
VALUES
(
    'EMP002',
    'Sarah Johnson',
    'VACATION',
    2026,
    32.00,
    40.00,
    16.00,
    8.00,
    4.00,
    NOW(),
    'USER02'
),
(
    'EMP002',
    'Sarah Johnson',
    'SICK',
    2026,
    12.00,
    24.00,
    8.00,
    0.00,
    2.00,
    NOW(),
    'USER02'
);

---------------------------------------------------------------------