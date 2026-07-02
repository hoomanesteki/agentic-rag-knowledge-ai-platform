// Ontology for apparel_ecommerce. Labels here match entity_types in domain.yaml.
// A unique constraint on each label's primary id lets the graph load MERGE idempotently.

CREATE CONSTRAINT product_id IF NOT EXISTS
  FOR (p:Product) REQUIRE p.product_id IS UNIQUE;

CREATE CONSTRAINT store_id IF NOT EXISTS
  FOR (s:Store) REQUIRE s.store_id IS UNIQUE;

CREATE CONSTRAINT supplier_id IF NOT EXISTS
  FOR (sup:Supplier) REQUIRE sup.supplier_id IS UNIQUE;

CREATE CONSTRAINT review_id IF NOT EXISTS
  FOR (r:Review) REQUIRE r.id IS UNIQUE;

// Relationship vocabulary shared by templated Cypher and text-to-Cypher:
//   (:Supplier)-[:SUPPLIES]->(:Product)
//   (:Product)-[:SOLD_AT]->(:Store)      // derived from sales lines
//   (:Review)-[:MENTIONS]->(:Product)
