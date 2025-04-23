from pydantic import BaseModel

from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI

PROMPT = (
    "You are a helpful research assistant. Given a query, come up with a set of web searches "
    "to perform to best answer the query. Output between 5 and 20 terms to query for."
)


class WebSearchItem(BaseModel):
    reason: str
    "Your reasoning for why this search is important to the query."

    query: str
    "The search term to use for the web search."


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]
    """A list of web searches to perform to best answer the query."""

model = OpenAIChatCompletionsModel(
    model="llama3.2:latest",
    openai_client=AsyncOpenAI(base_url="http://localhost:11434/v1")
)

planner_agent = Agent(
    name="PlannerAgent",
    instructions=PROMPT,
    model=model,
    output_type=WebSearchPlan,
)
