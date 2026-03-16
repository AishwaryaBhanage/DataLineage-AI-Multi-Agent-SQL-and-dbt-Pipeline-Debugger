-- Layer 3: Final model — INTENTIONALLY BROKEN
-- Bug: uses 'amount' but upstream stg_orders has 'amount_total'
select
    customer_id,
    sum(amount) as total_revenue,      -- ERROR: 'amount' does not exist
    count(order_id) as order_count,
    max(order_date) as last_order_date
from {{ ref('stg_orders') }}
group by customer_id
