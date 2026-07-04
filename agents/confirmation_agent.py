# agents/confirmation_agent.py
import logging
import datetime
import sys
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from graph.state import ResearchState

# Configure logging
logger = logging.getLogger("ConfirmationAgent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ConfirmationObject(BaseModel):
    """
    Structured model for presenting a proposed high-consequence action to the user.
    """
    action: str = Field(..., description="The high-consequence action name (e.g., 'Send introduction request').")
    reason: str = Field(..., description="The academic motivation or purpose for initiating this action.")
    affected_faculty: List[str] = Field(default_factory=list, description="A list of faculty members affected by this action.")
    generated_content: str = Field(..., description="The draft content (e.g., draft email or proposal text) awaiting review.")


class ConfirmationDecision(BaseModel):
    """
    Structured model representing the final collected human approval decision.
    """
    action: str = Field(..., description="The action being reviewed.")
    status: str = Field(..., description="The outcome of the decision ('approved', 'rejected', 'modified', or 'pending').")
    edited_content: Optional[str] = Field(None, description="The approved content, containing any manual edits if modified.")
    approval_timestamp: Optional[str] = Field(None, description="ISO-8601 timestamp of when the decision was recorded.")
    reason: Optional[str] = Field(None, description="Comments, reasons for rejection, or modification notes.")


class ConfirmationAgent:
    """
    The Human-in-the-Loop Confirmation Agent.
    
    Responsible for:
    - Receiving proposed actions.
    - Formatting them into clean, structured confirmation views.
    - Presenting the Action, Reason, Affected Faculty, and Draft Content.
    - Waiting for and collecting the user's decision (APPROVE, REJECT, EDIT).
    
    Strict Design Goals:
    - Performs approval collection only.
    - NO Gemini reasoning (prevents loops, bias, and cost).
    - NO vector retrieval.
    - NO database updates (does not execute the action directly).
    """

    def build_confirmation_object(self, pending_action: Dict[str, Any]) -> ConfirmationObject:
        """
        Extracts and structures a proposed action into a typed ConfirmationObject.
        Handles missing fields gracefully by substituting safe defaults.
        """
        action = pending_action.get("action") or "Unknown Action"
        reason = pending_action.get("reason") or "No justification provided."
        
        # Handle affected faculty list gracefully
        raw_faculty = pending_action.get("affected_faculty") or []
        affected_faculty = []
        if isinstance(raw_faculty, list):
            for f in raw_faculty:
                if isinstance(f, dict):
                    affected_faculty.append(f.get("name") or "Unnamed Faculty")
                else:
                    affected_faculty.append(str(f))
        elif isinstance(raw_faculty, str):
            affected_faculty = [raw_faculty]

        generated_content = pending_action.get("generated_content") or "No content draft found."

        return ConfirmationObject(
            action=action,
            reason=reason,
            affected_faculty=affected_faculty,
            generated_content=generated_content
        )

    def process_decision(self, conf_obj: ConfirmationObject, state: ResearchState) -> ConfirmationDecision:
        """
        Presents the proposed action and collects the human decision.
        Supports both real-time interactive terminal prompts (TTY) and state-driven web environments.
        """
        # 1. Present the Action, Reason, Affected Faculty, and Generated Content
        logger.info("=== Presenting Proposed Action for Verification ===")
        logger.info(f"Action:            {conf_obj.action}")
        logger.info(f"Reason:            {conf_obj.reason}")
        logger.info(f"Affected Faculty:  {', '.join(conf_obj.affected_faculty)}")
        logger.info("--- Draft Content ---")
        logger.info(conf_obj.generated_content)
        logger.info("====================================================")

        # 2. Interactive Terminal Fallback (for local testing / CLI runs)
        if sys.stdin.isatty():
            print("\n" + "="*60)
            print(" HUMAN-IN-THE-LOOP APPROVAL REQUIRED")
            print("="*60)
            print(f"Action:            {conf_obj.action}")
            print(f"Reason:            {conf_obj.reason}")
            print(f"Affected Faculty:  {', '.join(conf_obj.affected_faculty)}")
            print("-" * 60)
            print("GENERATED CONTENT DRAFT:")
            print(conf_obj.generated_content)
            print("="*60)
            
            while True:
                choice = input("Enter decision [APPROVE (a) / REJECT (r) / EDIT (e)]: ").strip().upper()
                if choice in ['A', 'APPROVE']:
                    return ConfirmationDecision(
                        action=conf_obj.action,
                        status="approved",
                        edited_content=conf_obj.generated_content,
                        approval_timestamp=datetime.datetime.now().isoformat(),
                        reason="Approved via interactive terminal prompt."
                    )
                elif choice in ['R', 'REJECT']:
                    comment = input("Enter reason for rejection: ").strip()
                    return ConfirmationDecision(
                        action=conf_obj.action,
                        status="rejected",
                        edited_content=None,
                        approval_timestamp=datetime.datetime.now().isoformat(),
                        reason=comment or "Rejected via interactive terminal prompt."
                    )
                elif choice in ['E', 'EDIT']:
                    print("Enter modified content (type 'EOF' on a new line when done):")
                    lines = []
                    while True:
                        line = input()
                        if line == 'EOF':
                            break
                        lines.append(line)
                    edited_text = "\n".join(lines)
                    reason_edit = input("Enter reason for modification (optional): ").strip()
                    return ConfirmationDecision(
                        action=conf_obj.action,
                        status="modified",
                        edited_content=edited_text,
                        approval_timestamp=datetime.datetime.now().isoformat(),
                        reason=reason_edit or "Edited via interactive terminal prompt."
                    )
                else:
                    print("Invalid choice. Please enter a, r, or e.")

        # 3. State-Driven Web Decision Processor (LangGraph Context)
        raw_status = state.get("approval_status") or "pending"
        status_norm = str(raw_status).lower().strip()

        if status_norm in ["approved", "approve"]:
            return ConfirmationDecision(
                action=conf_obj.action,
                status="approved",
                edited_content=conf_obj.generated_content,
                approval_timestamp=datetime.datetime.now().isoformat(),
                reason="Approved by user."
            )
        elif status_norm in ["rejected", "reject"]:
            return ConfirmationDecision(
                action=conf_obj.action,
                status="rejected",
                edited_content=None,
                approval_timestamp=datetime.datetime.now().isoformat(),
                reason="Rejected by user."
            )
        elif status_norm in ["modified", "edit", "edited"]:
            # If modified, the user should pass the edited text via current_query or state
            edited_text = state.get("current_query") or conf_obj.generated_content
            return ConfirmationDecision(
                action=conf_obj.action,
                status="modified",
                edited_content=edited_text,
                approval_timestamp=datetime.datetime.now().isoformat(),
                reason="Modified and approved by user."
            )
        else:
            # Default to pending state
            return ConfirmationDecision(
                action=conf_obj.action,
                status="pending",
                edited_content=conf_obj.generated_content,
                approval_timestamp=None,
                reason="Awaiting user decision (APPROVE / REJECT / EDIT)."
            )


def confirmation_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph agent node representing the human-in-the-loop action approval gate.
    Ensures safe approval collection without executing actions or calling Gemini.
    """
    logger.info("[Agent Node] executing confirmation_agent_node...")
    
    pending_action = state.get("pending_action")
    if not pending_action:
        logger.warning("confirmation_agent_node executed but no pending_action was found.")
        # Graceful handling for missing action
        return {
            "approval_required": False,
            "approval_status": "approved",
            "tool_output": "No action pending. Bypass confirmation.",
            "error": None
        }

    agent = ConfirmationAgent()
    try:
        # 1. Structure the action context
        conf_obj = agent.build_confirmation_object(pending_action)
        
        # 2. Process the human feedback/decision
        decision = agent.process_decision(conf_obj, state)
        
        # 3. Prepare LangGraph state updates
        is_pending = decision.status == "pending"
        
        # Format exact Task 3 return dict to merge into state or logs
        decision_dict = {
            "action": decision.action,
            "status": decision.status,
            "edited_content": decision.edited_content,
            "approval_timestamp": decision.approval_timestamp,
            "reason": decision.reason
        }
        
        logger.info(f"Confirmation Decision Processed -> Status: {decision.status}")
        
        # We update the state indicators and record the output
        return {
            "approval_required": is_pending,
            "approval_status": decision.status,
            "tool_output": f"Action approval complete. Decision: {decision.status.upper()}",
            "retrieved_context": f"Action: {decision.action} | Status: {decision.status}",
            "error": None
        }
    except Exception as e:
        logger.error(f"ConfirmationAgent node failed: {e}")
        return {
            "error": f"ConfirmationAgentNode failed: {str(e)}"
        }
