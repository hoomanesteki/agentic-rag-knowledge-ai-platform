-- GENERATED from the apparel_ecommerce manifest by scripts/dbt_codegen.py. Do not edit by hand.
select trim(CAST("supplier_id" AS VARCHAR)) AS "supplier_id", trim(CAST("name" AS VARCHAR)) AS "name", trim(CAST("country" AS VARCHAR)) AS "country", trim(CAST("material" AS VARCHAR)) AS "material" from {{ ref('bronze_suppliers') }}
