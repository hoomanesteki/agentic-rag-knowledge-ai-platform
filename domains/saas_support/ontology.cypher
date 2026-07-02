// Ontology for saas_support. Labels here match entity_types in domain.yaml.
// A unique constraint on each label's primary id lets the graph load MERGE idempotently.

CREATE CONSTRAINT plan_id IF NOT EXISTS
  FOR (p:Plan) REQUIRE p.plan_id IS UNIQUE;

CREATE CONSTRAINT ticket_id IF NOT EXISTS
  FOR (t:Ticket) REQUIRE t.ticket_id IS UNIQUE;

CREATE CONSTRAINT article_id IF NOT EXISTS
  FOR (a:Article) REQUIRE a.id IS UNIQUE;

// Relationship vocabulary shared by templated Cypher and text-to-Cypher:
//   (:Ticket)-[:ON_PLAN]->(:Plan)
//   (:Article)-[:ABOUT]->(:Plan)
