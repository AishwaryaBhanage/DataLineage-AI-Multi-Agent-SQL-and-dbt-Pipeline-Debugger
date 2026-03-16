-- Layer 2: Staging model — cleans raw data
-- NOTE: column is named 'amount_total' here (not 'amount')
select
    order_id,
    customer_id,
    price           as amount_total,   -- <-- renamed from 'price' to 'amount_total'
    status,
    cast(order_date as date) as order_date
from {{ ref('raw_orders') }}
where status != 'returned'
