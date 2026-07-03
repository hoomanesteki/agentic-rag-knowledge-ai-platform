-- Governance test: a column declared as PII must never carry a raw value past silver. Every
-- non-null value has to be a masked pseudonym ('masked:' + hash). The test returns the offending
-- rows, so dbt passes only when there are none.
{% test is_masked(model, column_name) %}
select {{ column_name }}
from {{ model }}
where {{ column_name }} is not null
  and left(cast({{ column_name }} as varchar), 7) <> 'masked:'
{% endtest %}
