import logging
import aiohttp
import pandas as pd
import requests
from typing import List, Tuple, Dict, Optional
import asyncio
import os
import google.generativeai as genai
from dotenv import load_dotenv
from ai_services_api.services.centralized_repository.database_manager import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ExpertProcessor:
    def __init__(self, db: DatabaseManager, base_url: str):
        """Initialize ExpertProcessor."""
        self.db = db
        self.base_url = base_url
        self.session = None
        
        # Initialize Gemini
        load_dotenv()
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    async def analyze_expertise_with_gemini(self, expertise: List[str]) -> Tuple[List[str], List[str]]:
        """Use Gemini to analyze expertise and map to academic domains and fields"""
        try:
            # Create prompt for Gemini
            prompt = f"""
            Given the following expertise areas: {', '.join(expertise)}

            Map these to:
            1. Academic domains from this list only: Social Sciences, Physical Sciences, Health Sciences, Life Sciences
            2. Academic fields from this list only: Decision Sciences, Nursing, Engineering, Social Sciences, Health Professions, 
               Computer Science, Business Management and Accounting, Mathematics, Psychology, Immunology and Microbiology, 
               Medicine, Economics Econometrics and Finance, Agricultural and Biological Sciences, Environmental Science, 
               Arts and Humanities, Biochemistry Genetics and Molecular Biology, Neuroscience, Energy, Chemistry, 
               Earth and Planetary Sciences, Materials Science, Physics and Astronomy, Veterinary

            Return the results in exactly this format:
            DOMAINS: [comma-separated list of relevant domains]
            FIELDS: [comma-separated list of relevant fields]
            """

            # Generate response from Gemini
            response = await asyncio.to_thread(
                lambda: self.model.generate_content(prompt).text
            )

            # Parse response
            domains = []
            fields = []
            
            for line in response.split('\n'):
                if line.startswith('DOMAINS:'):
                    domains = [d.strip() for d in line.replace('DOMAINS:', '').strip('[]').split(',')]
                elif line.startswith('FIELDS:'):
                    fields = [f.strip() for f in line.replace('FIELDS:', '').strip('[]').split(',')]

            return domains, fields

        except Exception as e:
            logger.error(f"Error analyzing expertise with Gemini: {e}")
            return [], []

    async def get_expert_expertise(self) -> List[Dict]:
        """Get all experts with their expertise from database."""
        try:
            results = self.db.execute("""
                SELECT first_name, last_name, knowledge_expertise
                FROM experts_expert
                WHERE knowledge_expertise IS NOT NULL 
                AND knowledge_expertise != '[]'::jsonb
            """)
            return [
                {
                    'first_name': row[0],
                    'last_name': row[1],
                    'expertise': row[2] if isinstance(row[2], list) else eval(row[2])
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Error fetching experts: {e}")
            return []

    async def update_expert_fields(self, first_name: str, last_name: str, expertise: List[str]) -> bool:
        """Update expert fields using Gemini analysis."""
        try:
            # Get domains and fields from Gemini
            domains, fields = await self.analyze_expertise_with_gemini(expertise)
            
            if domains or fields:
                # Update the database with the new data
                self.db.execute("""
                    UPDATE experts_expert
                    SET domains = %s::TEXT[],
                        fields = %s::TEXT[]
                    WHERE first_name = %s AND last_name = %s
                    RETURNING id
                """, (
                    domains,  # List of domains
                    fields,  # List of fields
                    first_name,  # First name for the WHERE clause
                    last_name  # Last name for the WHERE clause
                ))
                
                logger.info(f"Updated fields and domains for {first_name} {last_name}")
                logger.info(f"Domains: {domains}")
                logger.info(f"Fields: {fields}")
                return True
            else:
                logger.warning(f"No domains or fields generated for {first_name} {last_name}")
                return False
            
        except Exception as e:
            logger.error(f"Error updating expert fields for {first_name} {last_name}: {e}")
            return False

    async def process_all_experts(self):
        """Process all experts in the database."""
        try:
            # Get all experts
            experts = await self.get_expert_expertise()
            logger.info(f"Found {len(experts)} experts to process")

            for expert in experts:
                first_name = expert['first_name']
                last_name = expert['last_name']
                expertise = expert['expertise']
                
                if expertise:
                    logger.info(f"Processing {first_name} {last_name} with expertise: {expertise}")
                    await self.update_expert_fields(first_name, last_name, expertise)
                    # Add a small delay to avoid hitting rate limits
                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error processing all experts: {e}")

    def close(self):
        """Close database connection."""
        if hasattr(self, 'db'):
            self.db.close()