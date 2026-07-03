-- GENERATED from the apparel_ecommerce manifest by scripts/dbt_codegen.py. Do not edit by hand.
select * from read_csv_auto('/Users/esteki/Desktop/MDS/Projects/agentic-rag-knowledge-ai-platform/domains/apparel_ecommerce/seed/structured/products.csv', header=true, all_varchar=true)
