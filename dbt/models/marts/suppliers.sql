-- GENERATED from the apparel_ecommerce manifest by scripts/dbt_codegen.py. Do not edit by hand.
select * from {{ ref('silver_suppliers') }}
