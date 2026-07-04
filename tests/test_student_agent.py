from agents.student_agent import StudentAgentResponse


def test_student_agent_response_exposes_status():
    response = StudentAgentResponse(
        query="Explainable AI",
        recommended_faculty=[],
        reasoning="No matches were available.",
        alternative_faculty=[],
        follow_up_questions=[],
    )

    assert response.status == "success"
