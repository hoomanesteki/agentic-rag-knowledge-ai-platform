// Ontology for this domain. Labels here must match entity_types in domain.yaml.
// Declare a unique constraint on each label's primary id so the graph load can MERGE
// idempotently. Add indexes for anything you filter or hop on often.

// Constraints (also create backing indexes)
CREATE CONSTRAINT product_id IF NOT EXISTS
  FOR (p:Product) REQUIRE p.product_id IS UNIQUE;

CREATE CONSTRAINT store_id IF NOT EXISTS
  FOR (s:Store) REQUIRE s.store_id IS UNIQUE;

CREATE CONSTRAINT supplier_id IF NOT EXISTS
  FOR (sup:Supplier) REQUIRE sup.supplier_id IS UNIQUE;

CREATE CONSTRAINT review_id IF NOT EXISTS
  FOR (r:Review) REQUIRE r.id IS UNIQUE;

// Relationship types used by the graph load and the graph retriever:
//   (:Supplier)-[:SUPPLIES]->(:Product)
//   (:Product)-[:SOLD_AT]->(:Store)
//   (:Review)-[:MENTIONS]->(:Product)
//   (:Review)-[:HAS_ISSUE]->(:Issue)
// Relationship types do not need constraints, but document them here so text to Cypher
// and templated queries share one vocabulary.
