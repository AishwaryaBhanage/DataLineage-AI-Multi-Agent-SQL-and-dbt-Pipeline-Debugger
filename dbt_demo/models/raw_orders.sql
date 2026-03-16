-- Layer 1: Raw source data (simulates a warehouse table)
select 1        as order_id, 101 as customer_id, 250.00 as price, 'completed' as status, '2024-01-01' as order_date
union all
select 2,                    102,                300.00,           'completed',             '2024-01-02'
union all
select 3,                    101,                150.00,           'returned',              '2024-01-03'
union all
select 4,                    103,                500.00,           'completed',             '2024-01-04'
union all
select 5,                    102,                 75.00,           'pending',               '2024-01-05'
