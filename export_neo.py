import os
import logging
from datetime import datetime
import pandas as pd
from neo4j import GraphDatabase
from typing import List, Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Neo4jExporter:
    def __init__(self):
        """Initialize Neo4j exporter with connection details from environment."""
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD')
        self.export_dir = 'exports'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def create_export_directory(self) -> str:
        """Create timestamped export directory."""
        export_path = os.path.join(self.export_dir, f"neo4j_{self.timestamp}")
        os.makedirs(export_path, exist_ok=True)
        return export_path

    def get_all_node_labels(self) -> List[str]:
        """Get all node labels from the database."""
        query = """
        CALL db.labels() YIELD label
        RETURN collect(label) as labels
        """
        with self.driver.session() as session:
            result = session.run(query)
            return result.single()["labels"]

    def get_all_relationship_types(self) -> List[str]:
        """Get all relationship types from the database."""
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN collect(relationshipType) as types
        """
        with self.driver.session() as session:
            result = session.run(query)
            return result.single()["types"]

    def export_nodes(self, label: str, export_dir: str) -> int:
        """Export all nodes with a specific label to CSV."""
        try:
            query = f"""
            MATCH (n:{label})
            RETURN n
            """
            with self.driver.session() as session:
                result = session.run(query)
                records = []
                
                # Process each node
                for record in result:
                    node = record["n"]
                    node_dict = dict(node.items())
                    
                    # Handle nested structures
                    for key, value in node_dict.items():
                        if isinstance(value, (dict, list)):
                            node_dict[key] = json.dumps(value)
                    
                    records.append(node_dict)

                if records:
                    df = pd.DataFrame(records)
                    filename = os.path.join(export_dir, f"nodes_{label}.csv")
                    df.to_csv(filename, index=False)
                    logger.info(f"Exported {len(records)} {label} nodes to {filename}")
                    return len(records)
                else:
                    logger.warning(f"No nodes found for label: {label}")
                    return 0

        except Exception as e:
            logger.error(f"Error exporting {label} nodes: {e}")
            return 0

    def export_relationships(self, rel_type: str, export_dir: str) -> int:
        """Export all relationships of a specific type to CSV."""
        try:
            query = f"""
            MATCH ()-[r:{rel_type}]->()
            RETURN 
                id(startNode(r)) as source_id,
                labels(startNode(r)) as source_labels,
                id(endNode(r)) as target_id,
                labels(endNode(r)) as target_labels,
                r
            """
            with self.driver.session() as session:
                result = session.run(query)
                records = []
                
                for record in result:
                    rel_dict = {
                        'source_id': record['source_id'],
                        'source_labels': ','.join(record['source_labels']),
                        'target_id': record['target_id'],
                        'target_labels': ','.join(record['target_labels'])
                    }
                    
                    # Add relationship properties
                    rel_props = dict(record['r'].items())
                    for key, value in rel_props.items():
                        if isinstance(value, (dict, list)):
                            rel_props[key] = json.dumps(value)
                        elif isinstance(value, datetime):
                            rel_props[key] = value.isoformat()
                    rel_dict.update(rel_props)
                    
                    records.append(rel_dict)

                if records:
                    df = pd.DataFrame(records)
                    filename = os.path.join(export_dir, f"relationships_{rel_type}.csv")
                    df.to_csv(filename, index=False)
                    logger.info(f"Exported {len(records)} {rel_type} relationships to {filename}")
                    return len(records)
                else:
                    logger.warning(f"No relationships found for type: {rel_type}")
                    return 0

        except Exception as e:
            logger.error(f"Error exporting {rel_type} relationships: {e}")
            return 0

    def export_all(self):
        """Export all nodes and relationships from Neo4j."""
        try:
            self.connect()
            export_dir = self.create_export_directory()
            logger.info(f"Exporting to directory: {export_dir}")
            
            # Export statistics
            total_nodes = 0
            total_relationships = 0
            
            # Export nodes
            node_labels = self.get_all_node_labels()
            logger.info(f"Found {len(node_labels)} node types to export")
            
            for label in node_labels:
                count = self.export_nodes(label, export_dir)
                total_nodes += count
            
            # Export relationships
            rel_types = self.get_all_relationship_types()
            logger.info(f"Found {len(rel_types)} relationship types to export")
            
            for rel_type in rel_types:
                count = self.export_relationships(rel_type, export_dir)
                total_relationships += count
            
            # Export summary
            logger.info(f"""
            Export Summary:
            - Export directory: {export_dir}
            - Total node types exported: {len(node_labels)}
            - Total nodes exported: {total_nodes}
            - Total relationship types exported: {len(rel_types)}
            - Total relationships exported: {total_relationships}
            """)
            
            return export_dir

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
        finally:
            if self.driver:
                self.driver.close()
                logger.info("Neo4j connection closed")

def main():
    try:
        exporter = Neo4jExporter()
        export_dir = exporter.export_all()
        logger.info(f"Neo4j export completed successfully. Files saved in: {export_dir}")
    except Exception as e:
        logger.error(f"Neo4j export failed: {e}")
        raise

if __name__ == "__main__":
    main()