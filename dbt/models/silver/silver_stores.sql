-- GENERATED from the apparel_ecommerce manifest by scripts/dbt_codegen.py. Do not edit by hand.
select trim(CAST("store_id" AS VARCHAR)) AS "store_id", trim(CAST("name" AS VARCHAR)) AS "name", trim(CAST("city" AS VARCHAR)) AS "city", trim(CAST("country" AS VARCHAR)) AS "country" from {{ ref('bronze_stores') }}
