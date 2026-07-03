-- Single-schema project: every model lands in the target schema (main), never a custom suffix, so
-- gold tables are queryable as main.<role> exactly where the metric layer and graph loader look.
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ target.schema }}
{%- endmacro %}
