"""Neo4j database utilities."""

from typing import Optional
from neo4j import GraphDatabase
from shared.config import settings


def get_neo4j_driver():
    """Get a Neo4j driver instance."""
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def check_neo4j_health() -> bool:
    """Check if Neo4j is running and accessible."""
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True
    except Exception:
        return False


def run_cypher(query: str, parameters: Optional[dict] = None) -> list[dict]:
    """Run a Cypher query and return results."""
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    finally:
        driver.close()


def create_entities(entities: list[dict], chapter: str = "default"):
    """Create entity nodes in Neo4j.

    Each entity dict should have: name, type, properties (optional)
    """
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            for entity in entities:
                name = entity["name"]
                entity_type = entity.get("type", "Entity")
                properties = entity.get("properties", {})
                properties["chapter"] = chapter

                props_str = ", ".join(
                    f"e.{k} = ${k}" for k in properties.keys()
                )

                query = f"""
                MERGE (e:{entity_type} {{name: $name}})
                SET {props_str}
                RETURN e
                """
                params = {"name": name, **properties}
                session.run(query, params)
    finally:
        driver.close()


def create_relationships(relationships: list[dict], chapter: str = "default"):
    """Create relationships between entities in Neo4j.

    Each relationship dict should have: source, target, type, properties (optional)
    """
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            for rel in relationships:
                source = rel["source"]
                target = rel["target"]
                rel_type = rel.get("type", "RELATED_TO")
                properties = rel.get("properties", {})
                properties["chapter"] = chapter

                props_str = ""
                if properties:
                    props_str = "SET " + ", ".join(
                        f"r.{k} = ${k}" for k in properties.keys()
                    )

                query = f"""
                MATCH (a {{name: $source}})
                MATCH (b {{name: $target}})
                MERGE (a)-[r:{rel_type}]->(b)
                {props_str}
                RETURN r
                """
                params = {"source": source, "target": target, **properties}
                session.run(query, params)
    finally:
        driver.close()


def get_graph_data(chapter: Optional[str] = None) -> dict:
    """Get all nodes and relationships for visualization."""
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            # Get nodes
            if chapter:
                node_result = session.run(
                    "MATCH (n) WHERE n.chapter = $chapter RETURN n",
                    {"chapter": chapter},
                )
            else:
                node_result = session.run("MATCH (n) RETURN n")

            nodes = []
            for record in node_result:
                node = record["n"]
                nodes.append({
                    "id": node.element_id,
                    "labels": list(node.labels),
                    "properties": dict(node),
                })

            # Get relationships
            if chapter:
                rel_result = session.run(
                    """
                    MATCH (a)-[r]->(b)
                    WHERE r.chapter = $chapter
                    RETURN a.name AS source, type(r) AS type, b.name AS target, properties(r) AS props
                    """,
                    {"chapter": chapter},
                )
            else:
                rel_result = session.run(
                    """
                    MATCH (a)-[r]->(b)
                    RETURN a.name AS source, type(r) AS type, b.name AS target, properties(r) AS props
                    """
                )

            relationships = []
            for record in rel_result:
                relationships.append({
                    "source": record["source"],
                    "type": record["type"],
                    "target": record["target"],
                    "properties": dict(record["props"]) if record["props"] else {},
                })

            return {"nodes": nodes, "relationships": relationships}
    finally:
        driver.close()


def clear_chapter_graph(chapter: str):
    """Delete all nodes and relationships for a specific chapter."""
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            session.run(
                """
                MATCH (n)
                WHERE n.chapter = $chapter
                DETACH DELETE n
                """,
                {"chapter": chapter},
            )
    finally:
        driver.close()
