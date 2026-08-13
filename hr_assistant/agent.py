from langchain.agents import create_agent

from hr_assistant import config

def create_hr_agent(llm, tools):
    """Create and return a LangChain agent for HR policy assistance."""
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=config.SYSTEM_PROMPT
    )   

