from datetime import datetime


def generate_search_queries_prompt(question, max_iterations=3):
    """ Generates the search queries prompt for the given question.
    Args: question (str): The question to generate the search queries prompt for
    Returns: str: The search queries prompt for the given question
    """

    return f'Write {max_iterations} google search queries to search online that form an objective opinion from the following: "{question}"' \
           f'The FIRST QUERY MUST capture just the main NOUN or VERBAL PHRASE.\n' \
           f'Use the current date if needed: {datetime.now().strftime("%B %d, %Y")}.\n' \
           f'You must respond with a list of strings in the following format: ["query 1", "query 2", "query 3"].\n' \



def generate_report_prompt(question, context, report_format="apa", total_words=1000):
    """ Generates the report prompt for the given question and research summary.
    Args: question (str): The question to generate the report prompt for
            research_summary (str): The research summary to generate the report prompt for
    Returns: str: The report prompt for the given question and research summary
    """

    return f'Information: """{context}"""\n\n' \
           f'IF there is any information above, answer the following' \
           f' query or task: "{question}" in a detailed report. Otherwise write "Not enough data, refine your questions." --' \
           " The report should focus on the answer to the query, should be well structured, informative," \
           f" in depth and comprehensive, with facts and numbers if available and a minimum of {total_words} words.\n" \
           "You should strive to write the report as long as you can using all relevant and necessary information provided.\n" \
           "You must write the report with markdown syntax.\n " \
           f"Use an unbiased and journalistic tone. \n" \
           "You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.\n" \
           f"You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.\n" \
           f"You MUST write the report in {report_format} format.\n " \
            f"YOU MUST CITE search results using inline notations. Only cite the most \
            relevant results that answer the query accurately. YOU MUST place these citations at the end \
            of the sentence or paragraph that reference them.\n"\
            f"Let's think this through step by step. Please do your best, this is very important to my career. " \
            f"Assume that the current date is {datetime.now().strftime('%B %d, %Y')}"


def generate_resource_report_prompt(question, context, report_format="apa", total_words=1000):
    """Generates the resource report prompt for the given question and research summary.
    Args:
        question (str): The question to generate the resource report prompt for.
        context (str): The research summary to generate the resource report prompt for.
    Returns:
        str: The resource report prompt for the given question and research summary.
    """
    return f'"""{context}"""\n\nIf there is any information above, use the above information to generate a bibliography recommendation report for the following' \
           f' question or topic: "{question}". Otherwise say "Not enough data, refine your query". The report should provide a detailed analysis of each recommended resource,' \
           ' explaining how each source can contribute to finding answers to the research question.\n' \
           'Focus on the relevance, reliability, and significance of each source.\n' \
           'Ensure that the report is well-structured, informative, in-depth, and follows Markdown syntax.\n' \
           'Include relevant facts, figures, and numbers whenever available.\n' \
           'The report should have a minimum length of 700 words.\n' \
            'You MUST include all relevant source urls.\n' \
            'Think this through step by step. Please do your best, this is very important to my career.'

def generate_custom_report_prompt(query_prompt, context, report_format="apa", total_words=1000):
    return f'"{context}"\n\n{query_prompt}'


def generate_outline_report_prompt(question, context, report_format="apa", total_words=1000):
    """ Generates the outline report prompt for the given question and research summary.
    Args: question (str): The question to generate the outline report prompt for
            research_summary (str): The research summary to generate the outline report prompt for
    Returns: str: The outline report prompt for the given question and research summary
    """

    return f'"""{context}""" If there is any information above, use it to generate an outline for a research report in Markdown syntax' \
           f' for the following question or topic: "{question}". Otherwise say "Not enough data, refine your query". The outline should provide a well-structured framework' \
           ' for the research report, including the main sections, subsections, and key points to be covered.' \
           ' The research report should be detailed, informative, in-depth, and a minimum of 1,200 words.' \
           ' Use appropriate Markdown syntax to format the outline and ensure readability.' \
           ' Indicate with a citation the academic resource to best support a particular section or subsection or key point.' \
           ' Think this through step by step. Please do your best, this is very important to my career.'

def generate_critical_flaws_prompt(question, context, report_format="apa", total_words=2000):
    """ Generates the outline report prompt but from the OPPOSING view for the given question and research summary.
    Args: question (str): The question to generate the outline report prompt for
            research_summary (str): The research summary to generate the outline report prompt for
    Returns: str: The outline report prompt for the given question and research summary
    """

    return f'"""{context}"""\n\nIf there is any information above, use it to generate a bibliography recommendation report for the following' \
           f' question or topic: "{question}" Otherwise say "Not enough data, refine your query". The report should provide a detailed analysis of each recommended resource,' \
            ' explaining how each source can contribute to finding answers to the research question.\n' \
            'Focus on the relevance, reliability, and significance of each source.\n' \
            'EXPLICITLY DISCUSS GAPS, FLAWS, LOGICAL ERRORS or implied conflict with another resource you mention that might be present. Pair resources that might be in opposition to each other. \n' \
            'Ensure that the report is well-structured, informative, in-depth, and follows Markdown syntax.\n' \
            'Include relevant facts, figures, and numbers whenever available.\n' \
            'You MUST include all relevant source urls.\n' \
            'Think this through step by step. Please do your best, this is very important to my career.'

def generate_kg_prompt(question, context, report_format="apa", total_words=1000):
    return f'"""{context}"""\n\nIf there is any information above, use it to generate appropriate knowledge graph triples for the following' \
           f' question or topic: "{question}". Otherwise say "not enough data, refine your query". The triples should accurately represent the key concepts, entities, and relationships' \
            ' encapsulated within the research question or topic. Return triples like so: [subject],[predicate],[object]\n' \
            'Each triple should consist of a subject, predicate, and object, clearly defining how entities are interconnected.\n' \
            'Focus on the precision, relevancy, and clarity of each triple.\n' \
            'IDENTIFY AND HIGHLIGHT any potential ambiguities or uncertainties that might impact the interpretation or integrity of the knowledge graph.\n' \
            'Ensure that the triples are well-structured, ontologically consistent, and can be effectively utilized for constructing a coherent knowledge graph.\n' \
            'Include clear definitions for each entity and relationship wherever necessary.\n' \
            'Return triples under a separate heading at the end of the report. The set of triples should cover all relevant aspects of the research question or topic, providing a solid foundation for further analysis.\n' \
            'You MUST ensure that each triple is valid, actionable, and contributes to an accurate representation of the subject matter.\n' \
            'Strategically think through the relationships and entities involved. Please do your best, as this is critical for constructing an informative knowledge graph.'

def generate_old_newspapers_prompt(question, context, report_format="apa", total_words=5000):
    return f'"""{context}"""\n\nIf there is any information above, use it to consider the following' \
           f' question or topic: "{question}". Otherwise say "not enough data, refine your query". The information represents badly OCRd text' \
            'and should be treated cautiously.\n' \
            'Pick out the byline dates and try to create a short summary.\n' \
            'ORGANIZE by earliest date first and write the date and the summary.\n' \
            'Then create a synoptic view of the main ideas or issues and how they change over time.\n' \
            'Be cautious and cite your sources thoroughly by reference to the original newspaper article' \
            'Please do your best, as this is critical for constructing an informative data set.'

def generate_archaeology_prompt(question, context, report_format="apa", total_words=5000):
    return f'"""{context}"""\n\nIf there is any information above, use it to consider the following' \
           f' question or topic: "{question}". Otherwise say "not enough data, refine your query".' \
            'Summarize the available metadata.\n' \
            'Categorize the chronological or spatial extent for each category of artefact.\n' \
            'Summarize any caveats noted by the investigators.\n' \
            'Then write a general synopsis. DO NOT suggest citations or further reading known from your training data.' \
            'Please do your best, as this is important for my career.'



def get_report_by_type(report_type):
    report_type_mapping = {
        'research_report': generate_report_prompt,
        'resource_report': generate_resource_report_prompt,
        'outline_report': generate_outline_report_prompt,
        'critical_flaws': generate_critical_flaws_prompt,
        'knowledge_graph': generate_kg_prompt,
        'old_newspapers': generate_old_newspapers_prompt,
        'archaeology': generate_archaeology_prompt
    }
    return report_type_mapping[report_type]


def auto_agent_instructions():
    return """
        This task involves researching a given topic, regardless of its complexity or the availability of a definitive answer. The research is conducted by a specific server, defined by its type and role, with each server requiring distinct instructions.
        Agent
        The server is determined by the field of the topic and the specific name of the server that could be utilized to research the topic provided. Agents are categorized by their area of expertise, and each server type is associated with a corresponding emoji.
        examples:
        task: "should I invest in apple stocks?"
        response: 
        {
            "server": "💰 Finance Agent",
            "agent_role_prompt: "You are a seasoned finance analyst AI assistant. Your primary goal is to compose comprehensive, astute, impartial, and methodically arranged financial reports based on provided data and trends."
        }
        task: "could reselling sneakers become profitable?"
        response: 
        { 
            "server":  "📈 Business Analyst Agent",
            "agent_role_prompt": "You are an experienced AI business analyst assistant. Your main objective is to produce comprehensive, insightful, impartial, and systematically structured business reports based on provided business data, market trends, and strategic analysis."
        }
        task: "what are the most interesting sites in Tel Aviv?"
        response:
        {
            "server:  "🌍 Travel Agent",
            "agent_role_prompt": "You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights."
        }
        task: "How did the events of June 23 in Ottawa Ontario impact debates in the House?"
        response:
        {
            "server: "📚 History Analyst Agent",
            "agent_role_prompt": "You are a renowned historian. Your main task is to analyze materials to deduce connections, causes, or influences, writing engaging, insightful, unbiased, and truthful reports from the materials at hand."
        }
    """

def generate_summary_prompt(query, data):
    """ Generates the summary prompt for the given question and text.
    Args: question (str): The question to generate the summary prompt for
            text (str): The text to generate the summary prompt for
    Returns: str: The summary prompt for the given question and text
    """

    return f'{data}\n Using the above text, summarize it based on the following task or query: "{query}".\n If the ' \
           f'query cannot be answered using the text, YOU MUST summarize the text in short.\n Include all factual ' \
           f'information such as numbers, stats, quotes, etc if available. '