import asyncio
import csv
import logging
from typing import List
from tqdm import tqdm

from ai_services_api.services.recommendation.services.expert_service import ExpertsService
from ai_services_api.services.recommendation.core.database import GraphDatabase

class DataLoader:
    def __init__(self):
        self.expert_service = ExpertsService()
        self.graph = GraphDatabase()
        self.logger = logging.getLogger(__name__)

    async def load_initial_experts(self, orcid_file_path: str, batch_size: int = 50):
        """
        Load initial experts from a CSV file with comprehensive error handling and progress tracking
        
        Args:
            orcid_file_path (str): Path to the CSV file containing ORCIDs
            batch_size (int): Number of experts to process in each batch
        """
        try:
            # Read ORCIDs from CSV
            with open(orcid_file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                orcids = [row[0].strip() for row in reader if row]

            self.logger.info(f"Total experts to process: {len(orcids)}")

            # Process ORCIDs in batches with progress tracking
            for i in range(0, len(orcids), batch_size):
                batch = orcids[i:i + batch_size]
                
                # Use tqdm for progress visualization
                with tqdm(total=len(batch), desc=f"Processing Batch {i//batch_size + 1}") as pbar:
                    async def process_expert(orcid):
                        try:
                            result = await self.expert_service.add_expert(orcid)
                            pbar.update(1)
                            return result
                        except Exception as e:
                            self.logger.error(f"Error processing ORCID {orcid}: {e}")
                            pbar.update(1)
                            return None

                    # Use asyncio.gather for concurrent processing
                    batch_results = await asyncio.gather(
                        *[process_expert(orcid) for orcid in batch],
                        return_exceptions=True
                    )

                # Optionally log batch processing summary
                successful = sum(1 for result in batch_results if result is not None)
                self.logger.info(f"Batch {i//batch_size + 1}: {successful}/{len(batch)} experts processed successfully")

            self.logger.info("Initial data load complete!")

        except Exception as e:
            self.logger.error(f"Error in initial data load: {e}")
            raise

    def verify_graph(self):
        """
        Comprehensive graph verification with detailed statistics and logging
        
        Returns:
            dict: Detailed graph statistics
        """
        try:
            stats = self.graph.get_graph_stats()

            # Detailed logging of graph statistics
            self.logger.info("Graph Verification Results:")
            for stat_name, stat_value in stats.items():
                self.logger.info(f"{stat_name.replace('_', ' ').title()}: {stat_value}")

            # Additional verification checks
            if stats['expert_count'] == 0:
                self.logger.warning("Warning: No expert nodes found in the graph")
            
            if stats['relationship_count'] == 0:
                self.logger.warning("Warning: No relationships found in the graph")

            return stats

        except Exception as e:
            self.logger.error(f"Error verifying graph: {e}")
            return {}

    async def update_existing_experts(self, days_since_last_update: int = 30):
        """
        Update existing experts in the graph
        
        Args:
            days_since_last_update (int): Number of days since last update to trigger re-processing
        """
        try:
            # Query for experts that need updating (placeholder logic)
            query = """
            MATCH (e:Expert)
            WHERE NOT EXISTS(e.last_updated) 
               OR e.last_updated < datetime() - duration({days: $days})
            RETURN e.orcid AS orcid
            """
            
            # Fetch experts needing update
            experts_to_update = self.graph.query_graph(query, {'days': days_since_last_update})
            
            self.logger.info(f"Identified {len(experts_to_update)} experts for update")

            # Batch update with progress tracking
            for batch in [experts_to_update[i:i+50] for i in range(0, len(experts_to_update), 50)]:
                async def update_expert(record):
                    orcid = record['orcid']
                    try:
                        await self.expert_service.add_expert(orcid)
                        return orcid
                    except Exception as e:
                        self.logger.error(f"Error updating expert {orcid}: {e}")
                        return None

                await asyncio.gather(*[update_expert(record) for record in batch])

        except Exception as e:
            self.logger.error(f"Error in expert update process: {e}")

# Example usage
async def main():
    loader = DataLoader()
    
    # Load initial experts
    await loader.load_initial_experts('path/to/orcids.csv')
    
    # Verify graph
    stats = loader.verify_graph()
    
    # Update existing experts
    await loader.update_existing_experts()

if __name__ == "__main__":
    asyncio.run(main())