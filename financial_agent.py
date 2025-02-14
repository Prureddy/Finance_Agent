from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
load_dotenv()

## Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Gather and synthesize accurate and comprehensive information from the web",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=[
        "Search the web for getting the data"
    ],
    show_tools_calls=True,
    markdown=True,
)

## Financial Agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        ),
    ],
    instructions=[
        "Always include Sources","Use proper citations", "Mention current Stock Price"
    ],
    show_tool_calls=True,
    markdown=True,
)

## Multi AI Agent
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    instructions=[
        "Use Tables to Display the data",
        "Always include sources for all information to ensure transparency and credibility.",
        "Use tables and visual aids to present data in an accessible and organized manner, making it easier for users to digest complex information."
        "Based on the technical analysis give me the buying price"
    ],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response(
    "Summarize analyst recommendation and share the latest news for TATA Motors",
    stream=True
)

