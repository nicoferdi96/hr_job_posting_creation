#!/usr/bin/env python
from datetime import datetime
from typing import List, Literal, Optional

from crewai import Agent, LLM
from crewai.flow import Flow, listen, persist, router, start
from pydantic import BaseModel, Field
from crewai_tools import FirecrawlSearchTool

from hr_job_creation.crews.hr_crew.hr_crew import HrCrew


class Message(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class RoleInfo(BaseModel):
    job_role: Optional[str] = None
    location: Optional[str] = None
    company_name: Optional[str] = None


class RouterIntent(BaseModel):
    user_intent: Literal["job_creation", "conversation", "refinement"]
    role_info: Optional[RoleInfo] = Field(default_factory=RoleInfo)
    feedback: Optional[str] = None
    answer_message: Optional[str] = None
    reasoning: str


class FlowState(BaseModel):
    user_message: str = "HI, create a job posting for a data engineer based in NYC for Johnson & Johnson"
    message_history: List[Message] = []
    role_info: RoleInfo = Field(default_factory=RoleInfo)
    job_posting: Optional[str] = None
    feedback: Optional[str] = None
    answer_message: Optional[str] = None


@persist()
class HrJobCreationFlow(Flow[FlowState]):
    llm = LLM(model="gpt-5-nano")

    def add_message(self, role: str, content: str):
        """Add a message to the message history"""
        new_message = Message(role=role, content=content)
        self.state.message_history.append(new_message)

    @start()
    def starting_flow(self):
        if self.state.user_message:
            self.add_message("user", self.state.user_message)
        return self.state.user_message

    @router(starting_flow)
    def routing_intent(self):
        llm = LLM(model="gpt-5-nano", response_format=RouterIntent)

        has_posting = self.state.job_posting is not None

        prompt = f"""
        === TASK ===
        You are an intelligent router for an HR job creation assistant. Your job is to analyze
        the user's message and conversation history to extract job creation details and determine intent.

        === INSTRUCTIONS ===
        Extract any of these fields mentioned in the current message or conversation history:
        - **job_role**: The job title/role being created (e.g., "Software Engineer", "Marketing Manager")
        - **location**: The job location (e.g., "New York", "Remote", "London")
        - **company_name**: The company the job is for (e.g., "Google", "Acme Corp")

        **ALREADY COLLECTED VALUES (preserve these — do NOT set to null):**
        - job_role: {self.state.role_info.job_role or "Not yet collected"}
        - location: {self.state.role_info.location or "Not yet collected"}
        - company_name: {self.state.role_info.company_name or "Not yet collected"}

        **EXISTING JOB POSTING:** {"Yes — a posting has already been generated" if has_posting else "No posting yet"}

        **ROUTING RULES:**
        - Return "refinement" if a job posting ALREADY EXISTS and the user is giving feedback,
          requesting changes, or asking for improvements to the current posting.
          Also extract the **feedback** field: a concise summary of what the user wants changed.
        - Return "job_creation" if ALL THREE fields (job_role, location, company_name) are populated
          AND no job posting exists yet.
          Also return "job_creation" if the user wants a completely NEW posting for a different
          role/company (even if a posting exists — this resets state).
        - Return "conversation" if any field is still missing and no posting exists yet.

        === CONVERSATION REPLY (only when intent is "conversation") ===
        When the intent is "conversation", you MUST also generate a friendly reply in `answer_message`.
        This reply should:
        1. Respond naturally to the user's message
        2. Acknowledge information already collected from the ALREADY COLLECTED VALUES above
        3. Ask for any fields that are still "Not yet collected"
        4. Be warm, professional, and concise
        5. If the user hasn't mentioned anything about job creation yet, introduce yourself
           and explain that you can help create job postings

        For "job_creation" or "refinement" intents, set `answer_message` to null.

        === INPUT DATA ===
        **Current User Message:**
        {self.state.user_message}

        **Conversation History:**
        {self.state.message_history}

        === OUTPUT REQUIREMENTS ===
        1. **user_intent**: "job_creation", "conversation", or "refinement"
        2. **role_info**: An object with:
           - **job_role**: The job role if mentioned (or from already collected values)
           - **location**: The location if mentioned (or from already collected values)
           - **company_name**: The company name if mentioned (or from already collected values)
        3. **feedback**: If intent is "refinement", a concise summary of the requested changes. Otherwise null.
        4. **answer_message**: If intent is "conversation", a friendly reply to the user. Otherwise null.
        5. **reasoning**: Brief explanation of your decision
        """

        response = llm.call(prompt)

        # Merge newly extracted role fields into state (don't overwrite with None)
        self.state.role_info = response.role_info
        if response.feedback:
            self.state.feedback = response.feedback
        if response.answer_message:
            self.state.answer_message = response.answer_message

        return response.user_intent

    @listen("conversation")
    def follow_up_conversation(self):
        response = self.state.answer_message
        self.add_message("assistant", response)
        print(f"Assistant: {response}")
        return response

    @listen("job_creation")
    def handle_job_creation(self):

        crew = HrCrew().crew()
        result = crew.kickoff(
            inputs={
                "job_role": self.state.role_info.job_role,
                "location": self.state.role_info.location,
                "company_name": self.state.role_info.company_name,
            }
        )

        response = result.raw
        self.state.job_posting = response

        # Add the conversation response to history
        self.add_message("assistant", response)

        print(f"Conversation response: {response}")
        return response

    @listen("refinement")
    def handle_refinement(self):
        print(f"\nRefining job posting based on feedback: {self.state.feedback}\n")

        agent = Agent(
            role="Senior Job Posting Editor",
            goal="Refine and improve job postings based on specific feedback",
            backstory=(
                "You are an expert HR editor who specializes in polishing "
                "and refining job postings. You make precise, targeted changes "
                "based on feedback while preserving the overall quality and "
                "structure of the posting."
            ),
            tools=[FirecrawlSearchTool()],
            llm=self.llm,
            verbose=True,
        )

        prompt = f"""
        You have a job posting that needs refinement based on user feedback.

        === CURRENT JOB POSTING ===
        {self.state.job_posting}

        === USER FEEDBACK TO IMPLEMENTATION ===
        {self.state.feedback}
        
        Your task is to incorporate the feedback of the user and return the same job posting, in a markdown format,
        with the feedback incorporated. The feedback may vary but it will always be related with changes to the 
        job posting such as the removal or modification of sections, additional research on certain sections, etc.

        === INSTRUCTIONS ===
        - Make ONLY the changes requested in the feedback
        - Preserve the overall structure and quality of the posting
        - Keep all sections that aren't affected by the feedback
        - Return the complete updated job posting in markdown format
        """

        result = agent.kickoff(prompt)
        response = result.raw
        self.state.job_posting = response

        # Add the conversation response to history
        self.add_message("assistant", response)

        print(f"Conversation response: {response}")
        return response


def kickoff():
    hr_flow = HrJobCreationFlow(tracing=True)
    hr_flow.kickoff()


def plot():
    hr_flow = HrJobCreationFlow()
    hr_flow.plot()


if __name__ == "__main__":
    kickoff()
