def _fetch_experts_data(self):
    """Fetch experts data from PostgreSQL with enhanced data retrieval"""
    conn = None
    try:
        conn = self.get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                id,
                first_name, 
                last_name,
                knowledge_expertise,
                designation,
                theme,
                unit,
                orcid,
                domains,
                fields,
                subfields,
                is_active
            FROM experts_expert
            WHERE id IS NOT NULL
        """)
        
        experts_data = cur.fetchall()
        logger.info(f"Fetched {len(experts_data)} experts from database")
        return experts_data
    except Exception as e:
        logger.error(f"Error fetching experts data: {e}")
        return []
    finally:
        if conn:
            conn.close()

def create_expert_node(self, session, expert_data: tuple):
    """Create expert node with enhanced relationship structure"""
    try:
        # Unpack basic expert data
        (expert_id, first_name, last_name, knowledge_expertise, designation, 
         theme, unit, orcid, domains, fields, subfields, is_active) = expert_data
        
        expert_name = f"{first_name} {last_name}"

        # Create expert node with enhanced metadata
        session.run("""
            MERGE (e:Expert {id: $id})
            SET e.name = $name,
                e.designation = $designation,
                e.theme = $theme,
                e.unit = $unit,
                e.orcid = $orcid,
                e.is_active = $is_active,
                e.updated_at = datetime()
        """, {
            "id": str(expert_id),
            "name": expert_name,
            "designation": designation,
            "theme": theme,
            "unit": unit,
            "orcid": orcid,
            "is_active": is_active
        })

        # Create knowledge expertise relationships
        if knowledge_expertise:
            for expertise in knowledge_expertise:
                if expertise:
                    session.run("""
                        MERGE (ex:Expertise {name: $expertise})
                        MERGE (e:Expert {id: $expert_id})-[r:HAS_EXPERTISE]->(ex)
                        SET r.weight = 1.0,
                            r.last_updated = datetime()
                    """, {
                        "expert_id": str(expert_id),
                        "expertise": expertise
                    })

        # Create theme and unit relationships
        if theme:
            session.run("""
                MERGE (t:Theme {name: $theme})
                MERGE (e:Expert {id: $expert_id})-[r:BELONGS_TO_THEME]->(t)
                SET r.last_updated = datetime()
            """, {
                "expert_id": str(expert_id),
                "theme": theme
            })

        if unit:
            session.run("""
                MERGE (u:Unit {name: $unit})
                MERGE (e:Expert {id: $expert_id})-[r:BELONGS_TO_UNIT]->(u)
                SET r.last_updated = datetime()
            """, {
                "expert_id": str(expert_id),
                "unit": unit
            })

        # Create domain relationships with weights
        if domains:
            for domain in domains:
                if domain:
                    session.run("""
                        MERGE (d:Domain {name: $domain})
                        MERGE (e:Expert {id: $expert_id})-[r:HAS_DOMAIN]->(d)
                        SET r.weight = 1.0,
                            r.level = 'primary',
                            r.last_updated = datetime()
                    """, {
                        "expert_id": str(expert_id),
                        "domain": domain
                    })

        # Create field relationships with weights
        if fields:
            for field in fields:
                if field:
                    session.run("""
                        MERGE (f:Field {name: $field})
                        MERGE (e:Expert {id: $expert_id})-[r:HAS_FIELD]->(f)
                        SET r.weight = 0.7,
                            r.level = 'secondary',
                            r.last_updated = datetime()
                    """, {
                        "expert_id": str(expert_id),
                        "field": field
                    })

        # Create subfield relationships with weights if they exist
        if subfields:
            for subfield in subfields:
                if subfield:
                    session.run("""
                        MERGE (sf:Subfield {name: $subfield})
                        MERGE (e:Expert {id: $expert_id})-[r:HAS_SUBFIELD]->(sf)
                        SET r.weight = 0.5,
                            r.level = 'tertiary',
                            r.last_updated = datetime()
                    """, {
                        "expert_id": str(expert_id),
                        "subfield": subfield
                    })

        logger.info(f"Successfully created/updated expert node: {expert_name}")

    except Exception as e:
        logger.error(f"Error creating expert node for {expert_id}: {e}")
        raise

def _create_indexes(self):
    """Create enhanced indexes in Neo4j"""
    index_queries = [
        # Node indexes
        "CREATE INDEX expert_id IF NOT EXISTS FOR (e:Expert) ON (e.id)",
        "CREATE INDEX expert_name IF NOT EXISTS FOR (e:Expert) ON (e.name)",
        "CREATE INDEX expert_orcid IF NOT EXISTS FOR (e:Expert) ON (e.orcid)",
        "CREATE INDEX domain_name IF NOT EXISTS FOR (d:Domain) ON (d.name)",
        "CREATE INDEX field_name IF NOT EXISTS FOR (f:Field) ON (f.name)",
        "CREATE INDEX subfield_name IF NOT EXISTS FOR (sf:Subfield) ON (sf.name)",
        "CREATE INDEX expertise_name IF NOT EXISTS FOR (ex:Expertise) ON (ex.name)",
        "CREATE INDEX theme_name IF NOT EXISTS FOR (t:Theme) ON (t.name)",
        "CREATE INDEX unit_name IF NOT EXISTS FOR (u:Unit) ON (u.name)",
        
        # Fulltext indexes for search
        """CREATE FULLTEXT INDEX expert_fulltext IF NOT EXISTS 
           FOR (e:Expert) ON EACH [e.name, e.designation]""",
        """CREATE FULLTEXT INDEX expertise_fulltext IF NOT EXISTS 
           FOR (ex:Expertise) ON EACH [ex.name]"""
    ]
    
    with self._neo4j_driver.session() as session:
        for query in index_queries:
            try:
                session.run(query)
                logger.info(f"Index created: {query}")
            except Exception as e:
                logger.warning(f"Error creating index: {e}")