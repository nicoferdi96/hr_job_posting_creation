import os

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool
)


@CrewBase
class AiEnhancedJobPostingGeneratorCrew:
    """AiEnhancedJobPostingGenerator crew"""

    @agent
    def market_research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["market_research_analyst"],
            tools=[SerperDevTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,

            max_execution_time=None,
            llm=LLM(
                model="openai/gpt-4o-mini",
                temperature=0.7,
            ),

        )

    @agent
    def ai_tools_research_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["ai_tools_research_specialist"],

            tools=[			SerperDevTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,

            max_execution_time=None,
            llm=LLM(
                model="openai/gpt-4o-mini",
                temperature=0.7,
            ),

        )

    @agent
    def company_culture_analyst(self) -> Agent:

        return Agent(
            config=self.agents_config["company_culture_analyst"],


            tools=[				ScrapeWebsiteTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,

            max_execution_time=None,
            llm=LLM(
                model="openai/gpt-4o-mini",
                temperature=0.7,
            ),

        )

    @agent
    def ai_enhanced_job_posting_creator(self) -> Agent:

        return Agent(
            config=self.agents_config["ai_enhanced_job_posting_creator"],


            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,

            max_execution_time=None,
            llm=LLM(
                model="openai/gpt-4o-mini",
                temperature=0.7,
            ),

        )



    @task
    def analyze_job_market_landscape(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_job_market_landscape"],
            markdown=False,


        )

    @task
    def research_ai_tools_for_role(self) -> Task:
        return Task(
            config=self.tasks_config["research_ai_tools_for_role"],
            markdown=False,


        )

    @task
    def analyze_company_culture_and_brand(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_company_culture_and_brand"],
            markdown=False,


        )

    @task
    def generate_ai_enhanced_job_posting(self) -> Task:
        return Task(
            config=self.tasks_config["generate_ai_enhanced_job_posting"],
            markdown=False,


        )


    @crew
    def crew(self) -> Crew:
        """Creates the AiEnhancedJobPostingGenerator crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            chat_llm=LLM(model="openai/gpt-4o-mini"),
        )


